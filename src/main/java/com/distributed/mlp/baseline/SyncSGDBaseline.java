package com.distributed.mlp.baseline;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicInteger;

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.model.MathUtils;
import com.distributed.mlp.protocol.WeightSerializer;

/**
 * Synchronous SGD baseline using multiple worker threads coordinated by a CyclicBarrier.
 * Saves the final model to {@code results/sync_model.bin}.
 */
public final class SyncSGDBaseline {
    private static final int DEFAULT_WORKERS = 3;
    private static final int DEFAULT_EPOCHS = 3;
    private static final long DEFAULT_SEED = 42L;
    private static final double LEARNING_RATE = 1e-3;
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
        } catch (IOException | InterruptedException | BrokenBarrierException | RuntimeException e) {
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
                List<Sample> epochSamples = new ArrayList<>(dataset);
                Collections.shuffle(epochSamples, new Random(seed + epoch));

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
                                if (idx >= epochSamples.size()) {
                                    break;
                                }
                                Sample sample = epochSamples.get(idx);

                                double[] probs;
                                synchronized (modelLock) {
                                    probs = sharedModel.forward(sample.pixels());
                                    MLPModel.Gradient gradient = sharedModel.backward(sample.pixels(), sample.label());
                                    applyGradient(sharedModel, gradient, LEARNING_RATE);
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
        return model.toFlatWeights();
    }

    private static void applyGradient(MLPModel model, MLPModel.Gradient gradient, double learningRate) {
        double[] weights = flattenModelWeights(model);
        double[] delta = gradient.toFlatArray();
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * delta[i];
        }
        model.loadWeights(weights);
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