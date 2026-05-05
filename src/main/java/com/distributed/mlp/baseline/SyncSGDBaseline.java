package com.distributed.mlp.baseline;

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.model.MathUtils;
import com.distributed.mlp.protocol.WeightSerializer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Synchronous SGD baseline using multiple worker threads coordinated by a CyclicBarrier.
 * Saves the final model to {@code results/sync_model.bin}.
 */
public final class SyncSGDBaseline {
    private static final int DEFAULT_WORKERS = 3;
    private static final int DEFAULT_EPOCHS = 5;
    private static final long DEFAULT_SEED = 42L;
    private static final Path OUTPUT_CSV = Path.of("results", "sync_results.csv");
    private static final Path MODEL_PATH = Path.of("results", "sync_model.bin");

    private SyncSGDBaseline() {
    }

    public static void main(String[] args) {
        int workers = args.length >= 1 ? Integer.parseInt(args[0]) : DEFAULT_WORKERS;
        int epochs = args.length >= 2 ? Integer.parseInt(args[1]) : DEFAULT_EPOCHS;
        long seed = args.length >= 3 ? Long.parseLong(args[2]) : DEFAULT_SEED;

        try {
            run(workers, epochs, seed);
        } catch (Exception e) {
            System.err.println("SyncSGDBaseline failed: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    public static void run(int workers, int epochs, long seed)
            throws IOException, InterruptedException, BrokenBarrierException {
        if (workers <= 0) {
            throw new IllegalArgumentException("workers must be > 0");
        }
        if (epochs <= 0) {
            throw new IllegalArgumentException("epochs must be > 0");
        }

        System.out.println("[SyncSGDBaseline] Loading dataset...");
        DataLoader dataLoader = new DataLoader();
        List<Sample> dataset = dataLoader.loadShard(0, 1);
        if (dataset.isEmpty()) {
            throw new IOException("Dataset is empty. Check data/cifar-10-batches-bin/ path.");
        }
        System.out.printf("[SyncSGDBaseline] Loaded %,d samples%n", dataset.size());

        MLPModel sharedModel = new MLPModel();
        sharedModel.initXavier(seed);
        System.out.println("[SyncSGDBaseline] Model initialised (Xavier).");

        Files.createDirectories(OUTPUT_CSV.getParent());

        try (BufferedWriter writer = Files.newBufferedWriter(
                OUTPUT_CSV,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE)) {
            writer.write("epoch,loss,accuracy,wall_sec");
            writer.newLine();

            for (int epoch = 1; epoch <= epochs; epoch++) {
                long epochStart = System.nanoTime();
                int samplesPerWorker = Math.max(1, dataset.size() / workers);

                AtomicInteger nextIdx = new AtomicInteger(0);
                AtomicInteger correct = new AtomicInteger(0);
                AtomicInteger processed = new AtomicInteger(0);
                double[] totalLoss = new double[]{0.0};
                Object lossLock = new Object();
                Object modelLock = new Object();

                CyclicBarrier barrier = new CyclicBarrier(workers + 1);
                Thread[] workerThreads = new Thread[workers];

                for (int w = 0; w < workers; w++) {
                    workerThreads[w] = new Thread(() -> {
                        try {
                            for (int i = 0; i < samplesPerWorker; i++) {
                                int idx = nextIdx.getAndIncrement();
                                if (idx >= dataset.size()) {
                                    break;
                                }
                                Sample sample = dataset.get(idx);

                                double[] probs;
                                synchronized (modelLock) {
                                    probs = sharedModel.forward(sample.pixels());
                                    sharedModel.backward(sample.pixels(), sample.label());
                                }

                                int pred = argmax(probs);
                                if (pred == sample.label()) {
                                    correct.incrementAndGet();
                                }

                                synchronized (lossLock) {
                                    totalLoss[0] += MathUtils.crossEntropyLoss(probs, sample.label());
                                }
                                processed.incrementAndGet();
                            }
                            barrier.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        } catch (BrokenBarrierException e) {
                            throw new RuntimeException(e);
                        }
                    }, "sync-sgd-worker-" + w);
                    workerThreads[w].start();
                }

                barrier.await();
                for (Thread t : workerThreads) {
                    t.join();
                }

                int seen = Math.max(1, processed.get());
                double avgLoss = totalLoss[0] / seen;
                double accuracy = (double) correct.get() / seen;
                double wallSec = (System.nanoTime() - epochStart) / 1_000_000_000.0;

                writer.write(String.format(Locale.ROOT, "%d,%.6f,%.6f,%.6f", epoch, avgLoss, accuracy, wallSec));
                writer.newLine();
                writer.flush();

                System.out.printf(
                        Locale.ROOT,
                        "[SyncSGDBaseline] epoch=%d workers=%d loss=%.6f acc=%.4f wall_sec=%.3f%n",
                        epoch,
                        workers,
                        avgLoss,
                        accuracy,
                        wallSec);
            }
        }

        System.out.println("Sync baseline CSV written: " + OUTPUT_CSV.toAbsolutePath());

        // ---- Save the final model ----
        double[] weights = flattenModelWeights(sharedModel);
        byte[] serialized = WeightSerializer.toBytesDouble(weights);
        Files.write(MODEL_PATH, serialized);
        System.out.printf("[SyncSGDBaseline] Model saved to %s (%,d bytes)%n",
                MODEL_PATH.toAbsolutePath(), serialized.length);
    }

    /**
     * Uses reflection to flatten the private weight arrays of MLPModel.
     */
    private static double[] flattenModelWeights(MLPModel model) {
        try {
            java.lang.reflect.Field w1 = MLPModel.class.getDeclaredField("w1");
            java.lang.reflect.Field b1 = MLPModel.class.getDeclaredField("b1");
            java.lang.reflect.Field w2 = MLPModel.class.getDeclaredField("w2");
            java.lang.reflect.Field b2 = MLPModel.class.getDeclaredField("b2");
            java.lang.reflect.Field w3 = MLPModel.class.getDeclaredField("w3");
            java.lang.reflect.Field b3 = MLPModel.class.getDeclaredField("b3");

            w1.setAccessible(true);
            b1.setAccessible(true);
            w2.setAccessible(true);
            b2.setAccessible(true);
            w3.setAccessible(true);
            b3.setAccessible(true);

            double[][] W1 = (double[][]) w1.get(model);
            double[]   B1 = (double[])   b1.get(model);
            double[][] W2 = (double[][]) w2.get(model);
            double[]   B2 = (double[])   b2.get(model);
            double[][] W3 = (double[][]) w3.get(model);
            double[]   B3 = (double[])   b3.get(model);

            int total = MLPModel.INPUT_DIM  * MLPModel.HIDDEN1_DIM
                      + MLPModel.HIDDEN1_DIM
                      + MLPModel.HIDDEN1_DIM * MLPModel.HIDDEN2_DIM
                      + MLPModel.HIDDEN2_DIM
                      + MLPModel.HIDDEN2_DIM * MLPModel.OUTPUT_DIM
                      + MLPModel.OUTPUT_DIM;

            double[] flat = new double[total];
            int idx = 0;
            for (double[] row : W1) for (double v : row) flat[idx++] = v;
            for (double v : B1) flat[idx++] = v;
            for (double[] row : W2) for (double v : row) flat[idx++] = v;
            for (double v : B2) flat[idx++] = v;
            for (double[] row : W3) for (double v : row) flat[idx++] = v;
            for (double v : B3) flat[idx++] = v;

            return flat;
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException("Cannot access model weights – ensure MLPModel fields are accessible.", e);
        }
    }

    private static int argmax(double[] values) {
        int bestIdx = 0;
        double bestVal = values[0];
        for (int i = 1; i < values.length; i++) {
            if (values[i] > bestVal) {
                bestVal = values[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }
}