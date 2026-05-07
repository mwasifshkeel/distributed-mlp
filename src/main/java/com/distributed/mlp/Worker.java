package com.distributed.mlp;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.Socket;
import java.net.SocketException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
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
import com.distributed.mlp.optimisation.GradientCompressor;
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
    private static final int DEFAULT_IO_THREADS = 1;
    private static final int QUEUE_CAPACITY = 256;
    private static final int DEFAULT_MINI_BATCH_SIZE = 256;
    private static final Path LOG_PATH = Path.of("logs", "model.logs");
    private static volatile PrintStream consoleOut = System.out;
    private static volatile boolean fileLoggingEnabled = false;

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

        initLogging("worker-" + workerId);
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
        int miniBatchSize = resolveMiniBatchSize();
        int computeThreads = Integer.parseInt(System.getProperty(
            "mlp.computeThreads",
            String.valueOf(defaultComputeThreads(totalWorkers))));
        int ioThreads = Integer.parseInt(System.getProperty(
            "mlp.ioThreads",
            String.valueOf(DEFAULT_IO_THREADS)));
        System.out.printf("[Worker %d] Using %d compute thread(s), %d IO thread(s)%n",
            workerId, computeThreads, ioThreads);

        ExecutorService ioPool = Executors.newFixedThreadPool(ioThreads);
        ExecutorService computePool = Executors.newFixedThreadPool(1);
        ExecutorService batchPool = Executors.newFixedThreadPool(computeThreads);

        int totalSamplesToEnqueue = steps * miniBatchSize;
        AtomicInteger enqueuedSamples = new AtomicInteger(0);
        AtomicInteger nextBatch = new AtomicInteger(0);
        AtomicInteger completedBatches = new AtomicInteger(0);
        AtomicBoolean stopFlag = new AtomicBoolean(false);
        ReentrantLock socketLock = new ReentrantLock();

        boolean compressGradients = Boolean.parseBoolean(
            System.getProperty("mlp.compressGradients", "false"));
        int pullEvery = Integer.parseInt(System.getProperty("mlp.pullEvery", "10"));
        boolean verboseSamples = Boolean.parseBoolean(
            System.getProperty("mlp.verboseSamples", "false"));
        if (pullEvery <= 0) {
            throw new IllegalArgumentException("mlp.pullEvery must be > 0");
        }

        System.out.printf("[Worker %d] Connecting to %s:%d ...%n", workerId, host, port);
        try (Socket socket = new Socket(host, port); 
        DataInputStream in = new DataInputStream(socket.getInputStream()); 
        DataOutputStream out = new DataOutputStream(socket.getOutputStream())) {
            System.out.printf("[Worker %d] Connected. Starting IO and compute pools. steps=%d totalSamplesToEnqueue=%d%n",
                    workerId, steps, totalSamplesToEnqueue);
            // IO thread: just feeds the queue
            for (int i = 0; i < ioThreads; i++) {
                ioPool.submit(() -> {
                    System.out.printf("[Worker %d] IO thread started%n", workerId);
                    try {
                        while (!stopFlag.get()) {
                            int sampleIdx = enqueuedSamples.get();
                            if (sampleIdx >= totalSamplesToEnqueue) {
                                System.out.printf("[Worker %d] IO thread done (enqueued %d)%n",
                                        workerId, sampleIdx);
                                return;
                            }

                            Sample sample = shard.get(sampleIdx % shard.size());
                            boolean offered = sampleQueue.offer(sample, 200, TimeUnit.MILLISECONDS);
                            if (!offered) {
                                continue;
                            }

                            int newCount = enqueuedSamples.incrementAndGet();
                            if (newCount % 500 == 0) {
                                System.out.printf("[Worker %d] IO enqueued %d/%d (queue size=%d)%n",
                                        workerId, newCount, totalSamplesToEnqueue, sampleQueue.size());
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
                boolean weightsInitialized = false;

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
                            sampleQueue, stopFlag, enqueuedSamples, totalSamplesToEnqueue, miniBatchSize);

                        System.out.printf("[Worker %d] Batch %d assembled: %d samples (%.0fms)%n",
                                workerId, batchIdx + 1, miniBatch.size(),
                                (double) (System.currentTimeMillis() - batchStart));

                        if (miniBatch.isEmpty()) {
                            System.out.printf("[Worker %d] Empty mini-batch at step %d, skipping%n",
                                    workerId, batchIdx);
                            continue;
                        }

                        // Pull weights (lazy pull with always-on first pull)
                        if (!weightsInitialized || GradientCompressor.shouldPullWeights(batchIdx, pullEvery)) {
                            System.out.printf("[Worker %d] Batch %d: pulling weights...%n",
                                    workerId, batchIdx + 1);
                            socketLock.lock();
                            try {
                                sendPullRequest(out);
                                byte[] payload = readWeightResponseOrShutdown(in);
                                if (payload == null) {
                                    stopFlag.set(true);
                                    System.out.printf("[Worker %d] SHUTDOWN received at batch %d%n",
                                            workerId, batchIdx + 1);
                                    return;
                                }
                                double[] weights = decodeWeights(payload);
                                localModel.loadWeights(weights);
                                weightsInitialized = true;
                            } finally {
                                socketLock.unlock();
                            }
                            System.out.printf("[Worker %d] Batch %d: weights pulled (%.0fms so far)%n",
                                    workerId, batchIdx + 1,
                                    (double) (System.currentTimeMillis() - batchStart));
                        }

                        // Forward + backward
                        System.out.printf("[Worker %d] Batch %d: running forward/backward on %d samples...%n",
                                workerId, batchIdx + 1, miniBatch.size());
                        double[] batchGradient = computeBatchGradientParallel(
                            localModel, miniBatch, batchPool, computeThreads,
                            verboseSamples, workerId, batchIdx + 1);

                        double scale = 1.0 / miniBatch.size();
                        for (int g = 0; g < batchGradient.length; g++) {
                            batchGradient[g] *= scale;
                        }
                        System.out.printf("[Worker %d] Batch %d: forward/backward done (%.0fms so far)%n",
                                workerId, batchIdx + 1,
                                (double) (System.currentTimeMillis() - batchStart));

                        // Push gradient
                        byte[] payload = compressGradients
                            ? GradientCompressor.compressToFloat32(batchGradient)
                            : WeightSerializer.toBytesDouble(batchGradient);
                        System.out.printf("[Worker %d] Batch %d: pushing gradient (%d bytes)...%n",
                                workerId, batchIdx + 1, payload.length);
                        socketLock.lock();
                        try {
                            sendPushGradient(out, payload);
                        } finally {
                            socketLock.unlock();
                        }

                        int done = completedBatches.incrementAndGet();
                        if (fileLoggingEnabled) {
                            System.out.printf("[Worker %d] Batch %d/%d complete in %.0fms%n",
                                workerId, done, steps,
                                (double) (System.currentTimeMillis() - batchStart));
                        }
                        consoleOut.printf("[Worker %d] Batch %d/%d complete%n", workerId, done, steps);

                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    } catch (IOException | ProtocolException e) {
                        stopFlag.set(true);
                        if (isSocketClosed(e)) {
                            System.err.printf("[Worker %d] Socket closed during push/pull, exiting cleanly.%n",
                                    workerId);
                            return;
                        }
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
            batchPool.shutdown();
            boolean batchDone = batchPool.awaitTermination(1, TimeUnit.HOURS);
            if (!ioDone || !computeDone) {
                System.err.printf("[Worker %d] Pools timed out! ioDone=%s computeDone=%s batchDone=%s%n",
                        workerId, ioDone, computeDone, batchDone);
                stopFlag.set(true);
                ioPool.shutdownNow();
                computePool.shutdownNow();
                batchPool.shutdownNow();
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

    private static byte[] readWeightResponseOrShutdown(DataInputStream in) throws IOException, ProtocolException {
        int[] header = MessageProtocol.readHeader(in);
        int payloadLength = header[0] - MessageProtocol.HEADER_SIZE;
        int type = header[1];

        if (type == MessageProtocol.SHUTDOWN) {
            if (payloadLength > 0) {
                in.skipNBytes(payloadLength);
            }
            return null;
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
        return payload;
    }

    private static double[] decodeWeights(byte[] payload) {
        int expectedDouble = MLPModel.parameterCount();
        int doubleBytes = expectedDouble * Double.BYTES;
        int floatBytes = expectedDouble * Float.BYTES;

        if (payload.length == floatBytes) {
            return WeightSerializer.fromBytesFloat(payload);
        }
        if (payload.length == doubleBytes) {
            return WeightSerializer.fromBytesDouble(payload);
        }
        throw new IllegalArgumentException("Unexpected weight payload size: " + payload.length);
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
            int totalSamplesToEnqueue,
            int miniBatchSize) throws InterruptedException {
        List<Sample> miniBatch = new ArrayList<>(miniBatchSize);
        while (!stopFlag.get() && miniBatch.size() < miniBatchSize) {
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
        return gradient.toFlatArray();
    }

    private static double[] computeBatchGradientParallel(
            MLPModel model,
            List<Sample> miniBatch,
            ExecutorService batchPool,
            int computeThreads,
            boolean verboseSamples,
            int workerId,
            int batchNumber) throws Exception {

        int threads = Math.max(1, Math.min(miniBatch.size(), computeThreads));
        int chunkSize = (int) Math.ceil((double) miniBatch.size() / threads);

        List<java.util.concurrent.Callable<double[]>> tasks = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            int start = t * chunkSize;
            int end = Math.min(miniBatch.size(), start + chunkSize);
            if (start >= end) break;

            tasks.add(() -> {
                double[] partial = new double[MLPModel.parameterCount()];
                for (int i = start; i < end; i++) {
                    Sample sample = miniBatch.get(i);
                    if (verboseSamples) {
                        System.out.printf("[Worker %d] Batch %d: forward sample %d/%d%n",
                                workerId, batchNumber, i + 1, miniBatch.size());
                    }
                    model.accumulateBackward(sample.pixels(), sample.label(), partial);
                }
                return partial;
            });
        }

        List<java.util.concurrent.Future<double[]>> futures = batchPool.invokeAll(tasks);
        double[] total = null;
        for (java.util.concurrent.Future<double[]> f : futures) {
            double[] partial = f.get();
            if (partial == null) continue;
            if (total == null) {
                total = new double[partial.length];
            }
            for (int g = 0; g < partial.length; g++) {
                total[g] += partial[g];
            }
        }

        if (total == null) {
            total = new double[MLPModel.parameterCount()];
        }
        return total;
    }

    private static boolean isSocketClosed(Exception e) {
        if (e instanceof java.io.EOFException) {
            return true;
        }
        if (e instanceof SocketException) {
            return true;
        }
        String msg = e.getMessage();
        if (msg == null) return false;
        String lower = msg.toLowerCase(Locale.ROOT);
        return lower.contains("broken pipe")
                || lower.contains("connection reset")
                || lower.contains("socket closed");
    }

    private static int resolveMiniBatchSize() {
        String value = System.getProperty("mlp.miniBatch");
        if (value == null || value.isBlank()) {
            value = System.getenv("MLP_MINI_BATCH");
        }
        if (value == null || value.isBlank()) {
            return DEFAULT_MINI_BATCH_SIZE;
        }
        try {
            return Math.max(1, Integer.parseInt(value.trim()));
        } catch (NumberFormatException e) {
            return DEFAULT_MINI_BATCH_SIZE;
        }
    }

    private static int defaultComputeThreads(int totalWorkers) {
        int logicalCores = Runtime.getRuntime().availableProcessors();
        return Math.max(1, logicalCores / Math.max(1, totalWorkers));
    }

    private static void initLogging(String tag) {
        PrintStream stderr = System.err;
        try {
            Path logPath = resolveLogPath();
            if (logPath == null) {
                return;
            }
            Path parent = logPath.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            PrintStream fileOut = new PrintStream(new FileOutputStream(logPath.toFile(), true), true);
            consoleOut = System.out;
            System.setOut(fileOut);
            fileLoggingEnabled = true;
            System.out.printf("[Log] %s logging to %s%n", tag, logPath.toAbsolutePath());
        } catch (IOException e) {
            stderr.println("[Log] Failed to init file logging: " + e.getMessage());
        }
    }

    private static Path resolveLogPath() {
        String value = System.getProperty("mlp.logFile");
        if (value == null || value.isBlank()) {
            value = System.getenv("MLP_LOG_FILE");
        }
        if (value == null || value.isBlank()) {
            return LOG_PATH;
        }
        String trimmed = value.trim();
        if (trimmed.equalsIgnoreCase("off")) {
            return null;
        }
        return Path.of(trimmed);
    }
}
