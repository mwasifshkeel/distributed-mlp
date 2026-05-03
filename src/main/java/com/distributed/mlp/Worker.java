package com.distributed.mlp;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.model.MLPModel.Gradient;
import com.distributed.mlp.protocol.MessageProtocol;
import com.distributed.mlp.protocol.MessageProtocol.ProtocolException;
import com.distributed.mlp.protocol.WeightSerializer;

public final class Worker {

    private static final String DEFAULT_HOST = "127.0.0.1";
    private static final int DEFAULT_PORT = 9000;
    private static final int DEFAULT_WORKER_ID = 0;
    private static final int DEFAULT_TOTAL_WORKERS = 3;
    private static final int DEFAULT_STEPS = 100;
    private static final long DEFAULT_SEED = 42L;
    private static final int IO_THREADS = 1;
    private static final int QUEUE_CAPACITY = 256;
    private static final int MINI_BATCH_SIZE = 32;

    private final String host;
    private final int port;
    private final int workerId;
    private final int totalWorkers;
    private final int steps;
    private final long seed;

    private Worker(String host, int port, int workerId, int totalWorkers, int steps, long seed) {
        this.host = host;
        this.port = port;
        this.workerId = workerId;
        this.totalWorkers = totalWorkers;
        this.steps = steps;
        this.seed = seed;
    }

    public static void main(String[] args) {
        String host = args.length >= 1 ? args[0] : DEFAULT_HOST;
        int port = args.length >= 2 ? Integer.parseInt(args[1]) : DEFAULT_PORT;
        int workerId = args.length >= 3 ? Integer.parseInt(args[2]) : DEFAULT_WORKER_ID;
        int totalWorkers = args.length >= 4 ? Integer.parseInt(args[3]) : DEFAULT_TOTAL_WORKERS;
        int steps = args.length >= 5 ? Integer.parseInt(args[4]) : DEFAULT_STEPS;
        long seed = args.length >= 6 ? Long.parseLong(args[5]) : DEFAULT_SEED;

        Worker worker = new Worker(host, port, workerId, totalWorkers, steps, seed);
        try {
            worker.run();
        } catch (Exception e) {
            System.err.println("Worker failed: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    private void run() throws IOException, ProtocolException {
        System.out.printf("[Worker %d] Loading shard...%n", workerId);
        long t0 = System.currentTimeMillis();
        DataLoader loader = new DataLoader();
        List<Sample> shard = loader.loadShard(workerId, totalWorkers);
        System.out.printf("[Worker %d] Shard loaded: %d samples in %.1fs%n",
                workerId, shard.size(), (System.currentTimeMillis() - t0) / 1000.0);

        if (shard.isEmpty()) {
            throw new IOException("Worker shard is empty for workerId=" + workerId);
        }

        MLPModel model = new MLPModel();
        model.initXavier(seed + workerId);
        System.out.printf("[Worker %d] Model initialised. TOTAL_PARAMS=%d%n",
                workerId, MLPModel.INPUT_DIM * MLPModel.HIDDEN1_DIM + MLPModel.HIDDEN1_DIM
                + MLPModel.HIDDEN1_DIM * MLPModel.HIDDEN2_DIM + MLPModel.HIDDEN2_DIM
                + MLPModel.HIDDEN2_DIM * MLPModel.OUTPUT_DIM + MLPModel.OUTPUT_DIM);

        BlockingQueue<Sample> sampleQueue = new LinkedBlockingQueue<>(QUEUE_CAPACITY);
        // Force single compute thread to avoid nextBatch race exhausting steps instantly
        int computeThreads = 1;
        System.out.printf("[Worker %d] Using %d compute thread(s), %d IO thread(s)%n",
                workerId, computeThreads, IO_THREADS);

        ExecutorService ioPool = Executors.newFixedThreadPool(IO_THREADS);
        ExecutorService computePool = Executors.newFixedThreadPool(computeThreads);

        int totalSamplesToEnqueue = steps * MINI_BATCH_SIZE;
        AtomicInteger enqueuedSamples = new AtomicInteger(0);
        AtomicInteger nextBatch = new AtomicInteger(0);
        AtomicInteger completedBatches = new AtomicInteger(0);
        AtomicBoolean stopFlag = new AtomicBoolean(false);
        ReentrantLock socketLock = new ReentrantLock();

        System.out.printf("[Worker %d] Connecting to %s:%d ...%n", workerId, host, port);
        try (Socket socket = new Socket(host, port); 
        DataInputStream in = new DataInputStream(socket.getInputStream()); 
        DataOutputStream out = new DataOutputStream(socket.getOutputStream())) {
            System.out.printf("[Worker %d] Connected. Starting IO and compute pools. steps=%d totalSamplesToEnqueue=%d%n",
                    workerId, steps, totalSamplesToEnqueue);
            // IO thread: just feeds the queue
            for (int i = 0; i < IO_THREADS; i++) {
                ioPool.submit(() -> {
                    System.out.printf("[Worker %d] IO thread started%n", workerId);
                    try {
                        while (!stopFlag.get()) {
                            int sampleIdx = enqueuedSamples.getAndIncrement();
                            if (sampleIdx >= totalSamplesToEnqueue) {
                                System.out.printf("[Worker %d] IO thread done (enqueued %d)%n",
                                        workerId, sampleIdx);
                                return;
                            }
                            Sample sample = shard.get(sampleIdx % shard.size());
                            sampleQueue.put(sample);
                            if (sampleIdx % 500 == 0) {
                                System.out.printf("[Worker %d] IO enqueued %d/%d (queue size=%d)%n",
                                        workerId, sampleIdx, totalSamplesToEnqueue, sampleQueue.size());
                            }
                        }
                        System.out.printf("[Worker %d] IO thread stopped by stopFlag%n", workerId);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                });
            }

            // Compute thread
            computePool.submit(() -> {
                System.out.printf("[Worker %d] Compute thread started%n", workerId);
                MLPModel localModel = new MLPModel();
                localModel.initXavier(seed + workerId);

                while (!stopFlag.get()) {
                    int batchIdx = nextBatch.getAndIncrement();
                    if (batchIdx >= steps) {
                        System.out.printf("[Worker %d] Compute thread done (batchIdx=%d >= steps=%d)%n",
                                workerId, batchIdx, steps);
                        return;
                    }

                    System.out.printf("[Worker %d] Starting batch %d/%d (queue size=%d)%n",
                            workerId, batchIdx + 1, steps, sampleQueue.size());

                    try {
                        long batchStart = System.currentTimeMillis();

                        List<Sample> miniBatch = assembleMiniBatch(
                                sampleQueue, stopFlag, enqueuedSamples, totalSamplesToEnqueue);

                        System.out.printf("[Worker %d] Batch %d assembled: %d samples (%.0fms)%n",
                                workerId, batchIdx + 1, miniBatch.size(),
                                (double) (System.currentTimeMillis() - batchStart));

                        if (miniBatch.isEmpty()) {
                            System.out.printf("[Worker %d] Empty mini-batch at step %d, skipping%n",
                                    workerId, batchIdx);
                            continue;
                        }

                        // Pull weights
                        System.out.printf("[Worker %d] Batch %d: pulling weights...%n",
                                workerId, batchIdx + 1);
                        socketLock.lock();
                        try {
                            sendPullRequest(out);
                            boolean shutdown = readWeightResponseOrShutdown(in);
                            if (shutdown) {
                                stopFlag.set(true);
                                System.out.printf("[Worker %d] SHUTDOWN received at batch %d%n",
                                        workerId, batchIdx + 1);
                                return;
                            }
                        } finally {
                            socketLock.unlock();
                        }
                        System.out.printf("[Worker %d] Batch %d: weights pulled (%.0fms so far)%n",
                                workerId, batchIdx + 1,
                                (double) (System.currentTimeMillis() - batchStart));

                        // Forward + backward
                        System.out.printf("[Worker %d] Batch %d: running forward/backward on %d samples...%n",
                                workerId, batchIdx + 1, miniBatch.size());
                        double[] batchGradient = null;
                        int sampleNum = 0;
                        for (Sample sample : miniBatch) {
                            sampleNum++;
                            System.out.printf("[Worker %d] Batch %d: forward sample %d/%d%n",
                                    workerId, batchIdx + 1, sampleNum, miniBatch.size());
                            try {
                                double[] probs = localModel.forward(sample.pixels());
                                System.out.printf("[Worker %d] Batch %d: backward sample %d/%d%n",
                                        workerId, batchIdx + 1, sampleNum, miniBatch.size());
                                Gradient grad = localModel.backward(sample.pixels(), sample.label());
                                double[] flatGrad = flattenGradient(grad);
                                if (batchGradient == null) {
                                    batchGradient = new double[flatGrad.length];
                                }
                                for (int g = 0; g < flatGrad.length; g++) {
                                    batchGradient[g] += flatGrad[g];
                                }
                            } catch (Exception e) {
                                System.err.printf("[Worker %d] ERROR on sample %d (label=%d, pixels.length=%d): %s%n",
                                        workerId, sampleNum, sample.label(), sample.pixels().length, e.getMessage());
                                e.printStackTrace(System.err);
                                throw e;
                            }
                        }

                        double scale = 1.0 / miniBatch.size();
                        for (int g = 0; g < batchGradient.length; g++) {
                            batchGradient[g] *= scale;
                        }
                        System.out.printf("[Worker %d] Batch %d: forward/backward done (%.0fms so far)%n",
                                workerId, batchIdx + 1,
                                (double) (System.currentTimeMillis() - batchStart));

                        // Push gradient
                        byte[] payload = WeightSerializer.toBytesDouble(batchGradient);
                        System.out.printf("[Worker %d] Batch %d: pushing gradient (%d bytes)...%n",
                                workerId, batchIdx + 1, payload.length);
                        socketLock.lock();
                        try {
                            sendPushGradient(out, payload);
                        } finally {
                            socketLock.unlock();
                        }

                        int done = completedBatches.incrementAndGet();
                        System.out.printf("[Worker %d] ✓ Batch %d/%d complete in %.0fms%n",
                                workerId, done, steps,
                                (double) (System.currentTimeMillis() - batchStart));

                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    } catch (IOException | ProtocolException e) {
                        stopFlag.set(true);
                        System.err.printf("[Worker %d] IO/Protocol ERROR: %s%n", workerId, e.getMessage());
                        e.printStackTrace(System.err);
                        throw new RuntimeException(e);
                    } catch (Exception e) {
                        stopFlag.set(true);
                        System.err.printf("[Worker %d] UNEXPECTED ERROR: %s%n", workerId, e.getMessage());
                        e.printStackTrace(System.err);
                        throw new RuntimeException(e);
                    }
                }
                System.out.printf("[Worker %d] Compute thread exiting (stopFlag=%s)%n",
                        workerId, stopFlag.get());
            });

            ioPool.shutdown();
            computePool.shutdown();


            System.out.printf("[Worker %d] Waiting for pools to finish...%n", workerId);
            boolean ioDone = ioPool.awaitTermination(1, TimeUnit.HOURS);
            boolean computeDone = computePool.awaitTermination(1, TimeUnit.HOURS);
            if (!ioDone || !computeDone) {
                System.err.printf("[Worker %d] Pools timed out! ioDone=%s computeDone=%s%n",
                        workerId, ioDone, computeDone);
                stopFlag.set(true);
                ioPool.shutdownNow();
                computePool.shutdownNow();
            }
            System.out.printf("[Worker %d] All done. Completed %d batches.%n",
                    workerId, completedBatches.get());

        } catch (SocketException e) {
            System.err.printf("[Worker %d] Socket closed: %s%n", workerId, e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Worker interrupted", e);
        } catch (RuntimeException e) {
            Throwable cause = e.getCause();
            if (cause instanceof IOException io) {
                throw io;
            }
            if (cause instanceof ProtocolException pe) {
                throw pe;
            }
            throw e;
        }
    }

    private static void sendPullRequest(DataOutputStream out) throws IOException, ProtocolException {
        MessageProtocol.writeHeader(out, MessageProtocol.PULL_REQUEST, 0);
        out.flush();
    }

    private static boolean readWeightResponseOrShutdown(DataInputStream in) throws IOException, ProtocolException {
        int[] header = MessageProtocol.readHeader(in);
        int payloadLength = header[0] - MessageProtocol.HEADER_SIZE;
        int type = header[1];

        if (type == MessageProtocol.SHUTDOWN) {
            if (payloadLength > 0) {
                in.skipNBytes(payloadLength);
            }
            return true;
        }
        if (type != MessageProtocol.WEIGHT_RESPONSE) {
            if (payloadLength > 0) {
                in.skipNBytes(payloadLength);
            }
            throw new ProtocolException("Expected WEIGHT_RESPONSE/SHUTDOWN but got type=0x"
                    + Integer.toHexString(type));
        }
        byte[] payload = new byte[payloadLength];
        in.readFully(payload);
        return false;
    }

    private static void sendPushGradient(DataOutputStream out, byte[] gradientPayload)
            throws IOException, ProtocolException {
        MessageProtocol.writeHeader(out, MessageProtocol.PUSH_GRADIENT, gradientPayload.length);
        out.write(gradientPayload);
        out.flush();
    }

    private static List<Sample> assembleMiniBatch(
            BlockingQueue<Sample> sampleQueue,
            AtomicBoolean stopFlag,
            AtomicInteger enqueuedSamples,
            int totalSamplesToEnqueue) throws InterruptedException {
        List<Sample> miniBatch = new ArrayList<>(MINI_BATCH_SIZE);
        while (!stopFlag.get() && miniBatch.size() < MINI_BATCH_SIZE) {
            Sample sample = sampleQueue.poll(2, TimeUnit.SECONDS);
            if (sample != null) {
                miniBatch.add(sample);
                continue;
            }
            if (enqueuedSamples.get() >= totalSamplesToEnqueue && sampleQueue.isEmpty()) {
                break;
            }
        }
        return miniBatch;
    }

    private static double[] flattenGradient(Gradient gradient) {
        int size = MLPModel.INPUT_DIM * MLPModel.HIDDEN1_DIM
                + MLPModel.HIDDEN1_DIM
                + MLPModel.HIDDEN1_DIM * MLPModel.HIDDEN2_DIM
                + MLPModel.HIDDEN2_DIM
                + MLPModel.HIDDEN2_DIM * MLPModel.OUTPUT_DIM
                + MLPModel.OUTPUT_DIM;

        double[] flat = new double[size];
        int idx = 0;
        idx = flatten2D(gradient.getDW1(), flat, idx);
        idx = flatten1D(gradient.getDb1(), flat, idx);
        idx = flatten2D(gradient.getDW2(), flat, idx);
        idx = flatten1D(gradient.getDb2(), flat, idx);
        idx = flatten2D(gradient.getDW3(), flat, idx);
        flatten1D(gradient.getDb3(), flat, idx);
        return flat;
    }

    private static int flatten2D(double[][] src, double[] dst, int idx) {
        for (double[] row : src) {
            for (double value : row) {
                dst[idx++] = value;
            }
        }
        return idx;
    }

    private static int flatten1D(double[] src, double[] dst, int idx) {
        for (double value : src) {
            dst[idx++] = value;
        }
        return idx;
    }
}
