package com.distributed.mlp;

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.model.MLPModel.Gradient;
import com.distributed.mlp.protocol.MessageProtocol;
import com.distributed.mlp.protocol.MessageProtocol.ProtocolException;
import com.distributed.mlp.protocol.WeightSerializer;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.net.SocketException;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Worker entrypoint for distributed training.
 *
 * connect -> pull -> forward/backward -> push.
 */
public final class Worker {
    private static final String DEFAULT_HOST = "127.0.0.1";
    private static final int DEFAULT_PORT = 9000;
    private static final int DEFAULT_WORKER_ID = 0;
    private static final int DEFAULT_TOTAL_WORKERS = 3;
    private static final int DEFAULT_STEPS = 100;
    private static final long DEFAULT_SEED = 42L;
    private static final int IO_THREADS = 2;
    private static final int QUEUE_CAPACITY = 256;

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
        DataLoader loader = new DataLoader();
        List<Sample> shard = loader.loadShard(workerId, totalWorkers);
        if (shard.isEmpty()) {
            throw new IOException("Worker shard is empty for workerId=" + workerId);
        }

        MLPModel model = new MLPModel();
        model.initXavier(seed + workerId);

        BlockingQueue<Sample> sampleQueue = new LinkedBlockingQueue<>(QUEUE_CAPACITY);
        int computeThreads = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
        ExecutorService ioPool = Executors.newFixedThreadPool(IO_THREADS);
        ExecutorService computePool = Executors.newFixedThreadPool(computeThreads);

        AtomicInteger enqueuedSteps = new AtomicInteger(0);
        AtomicInteger nextStep = new AtomicInteger(0);
        AtomicInteger completedSteps = new AtomicInteger(0);
        AtomicBoolean stopFlag = new AtomicBoolean(false);
        ReentrantLock socketLock = new ReentrantLock();

        try (Socket socket = new Socket(host, port);
             DataInputStream in = new DataInputStream(socket.getInputStream());
             DataOutputStream out = new DataOutputStream(socket.getOutputStream())) {

            System.out.printf("Worker %d connected to %s:%d with shard=%d samples%n",
                    workerId,
                    host,
                    port,
                    shard.size());

            for (int i = 0; i < IO_THREADS; i++) {
                ioPool.submit(() -> {
                    try {
                        while (!stopFlag.get()) {
                            int stepIdx = enqueuedSteps.getAndIncrement();
                            if (stepIdx >= steps) {
                                return;
                            }
                            Sample sample = shard.get(stepIdx % shard.size());
                            sampleQueue.put(sample);
                        }
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                });
            }

            for (int i = 0; i < computeThreads; i++) {
                computePool.submit(() -> {
                    while (!stopFlag.get()) {
                        int step = nextStep.getAndIncrement();
                        if (step >= steps) {
                            return;
                        }

                        try {
                            Sample sample = sampleQueue.poll(2, TimeUnit.SECONDS);
                            if (sample == null) {
                                if (enqueuedSteps.get() >= steps) {
                                    return;
                                }
                                continue;
                            }

                            socketLock.lock();
                            try {
                                sendPullRequest(out);
                                boolean shutdown = readWeightResponseOrShutdown(in);
                                if (shutdown) {
                                    stopFlag.set(true);
                                    System.out.printf("Worker %d received SHUTDOWN at step %d%n", workerId, step);
                                    return;
                                }
                            } finally {
                                socketLock.unlock();
                            }

                            model.forward(sample.pixels());
                            Gradient grad = model.backward(sample.pixels(), sample.label());
                            byte[] payload = WeightSerializer.toBytesDouble(flattenGradient(grad));

                            socketLock.lock();
                            try {
                                sendPushGradient(out, payload);
                            } finally {
                                socketLock.unlock();
                            }

                            int done = completedSteps.incrementAndGet();
                            if (done % 10 == 0) {
                                System.out.printf("Worker %d completed step %d/%d%n", workerId, done, steps);
                            }
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return;
                        } catch (IOException | ProtocolException e) {
                            stopFlag.set(true);
                            throw new RuntimeException(e);
                        }
                    }
                });
            }

            ioPool.shutdown();
            computePool.shutdown();

            boolean ioDone = ioPool.awaitTermination(1, TimeUnit.HOURS);
            boolean computeDone = computePool.awaitTermination(1, TimeUnit.HOURS);
            if (!ioDone || !computeDone) {
                stopFlag.set(true);
                ioPool.shutdownNow();
                computePool.shutdownNow();
            }
        } catch (SocketException e) {
            System.err.printf("Worker %d socket closed: %s%n", workerId, e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Worker interrupted while awaiting thread pools", e);
        } catch (RuntimeException e) {
            Throwable cause = e.getCause();
            if (cause instanceof IOException ioException) {
                throw ioException;
            }
            if (cause instanceof ProtocolException protocolException) {
                throw protocolException;
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

    private static void sendPushGradient(DataOutputStream out, byte[] gradientPayload) throws IOException, ProtocolException {
        MessageProtocol.writeHeader(out, MessageProtocol.PUSH_GRADIENT, gradientPayload.length);
        out.write(gradientPayload);
        out.flush();
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