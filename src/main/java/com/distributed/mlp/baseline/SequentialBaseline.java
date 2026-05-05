package com.distributed.mlp.baseline;

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.model.MathUtils;
import com.distributed.mlp.protocol.WeightSerializer;    // <-- new import

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Locale;

/**
 * Single‑threaded baseline that iterates a full dataset shard (worker 0 of 1)
 * and writes per‑epoch timing/metrics to CSV.
 * <p>
 * After training, the model is saved to {@code results/sequential_model.bin}.
 */
public final class SequentialBaseline {
    private static final int DEFAULT_EPOCHS = 5;
    private static final long DEFAULT_SEED = 42L;
    private static final Path OUTPUT_CSV = Path.of("results", "sequential_results.csv");
    private static final Path MODEL_PATH = Path.of("results", "sequential_model.bin");

    private SequentialBaseline() {
    }

    public static void main(String[] args) {
        int epochs = args.length >= 1 ? Integer.parseInt(args[0]) : DEFAULT_EPOCHS;
        long seed = args.length >= 2 ? Long.parseLong(args[1]) : DEFAULT_SEED;

        try {
            run(epochs, seed);
        } catch (IOException e) {
            System.err.println("SequentialBaseline failed: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    public static void run(int epochs, long seed) throws IOException {
        if (epochs <= 0) {
            throw new IllegalArgumentException("epochs must be > 0");
        }

        System.out.println("[SequentialBaseline] Loading dataset...");
        DataLoader dataLoader = new DataLoader();
        List<Sample> dataset = dataLoader.loadShard(0, 1);
        if (dataset.isEmpty()) {
            throw new IOException("Dataset is empty. Check data/cifar-10-batches-bin/ path.");
        }
        System.out.printf("[SequentialBaseline] Loaded %,d samples%n", dataset.size());

        MLPModel model = new MLPModel();
        model.initXavier(seed);
        System.out.println("[SequentialBaseline] Model initialised (Xavier).");

        Files.createDirectories(OUTPUT_CSV.getParent());

        // Write per‑epoch CSV
        try (BufferedWriter writer = Files.newBufferedWriter(
                OUTPUT_CSV,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE)) {
            writer.write("epoch,loss,accuracy,wall_sec");
            writer.newLine();

            for (int epoch = 1; epoch <= epochs; epoch++) {
                long startNanos = System.nanoTime();
                double totalLoss = 0.0;
                int correct = 0;

                for (Sample sample : dataset) {
                    double[] probs = model.forward(sample.pixels());
                    int prediction = argmax(probs);
                    if (prediction == sample.label()) {
                        correct++;
                    }

                    totalLoss += MathUtils.crossEntropyLoss(probs, sample.label());

                    // Backward pass (kept to match compute flow)
                    model.backward(sample.pixels(), sample.label());
                }

                double avgLoss = totalLoss / dataset.size();
                double accuracy = (double) correct / dataset.size();
                double wallSec = (System.nanoTime() - startNanos) / 1_000_000_000.0;

                writer.write(String.format(
                        Locale.ROOT,
                        "%d,%.6f,%.6f,%.6f",
                        epoch,
                        avgLoss,
                        accuracy,
                        wallSec));
                writer.newLine();
                writer.flush();

                System.out.printf(
                        Locale.ROOT,
                        "[SequentialBaseline] epoch=%d loss=%.6f acc=%.4f wall_sec=%.3f%n",
                        epoch,
                        avgLoss,
                        accuracy,
                        wallSec);
            }
        }

        System.out.println("Sequential baseline CSV written: " + OUTPUT_CSV.toAbsolutePath());

        // -------- Save the final model parameters ----------
        double[] weights = flattenModelWeights(model);
        byte[] serialized = WeightSerializer.toBytesDouble(weights);
        Files.write(MODEL_PATH, serialized);
        System.out.printf("[SequentialBaseline] Model saved to %s (%,d bytes)%n",
                MODEL_PATH.toAbsolutePath(), serialized.length);
    }

    /**
     * Flattens all trainable parameters into a 1‑D double array in the
     * exact order expected by the distributed code:
     * W1, b1, W2, b2, W3, b3.
     */
    private static double[] flattenModelWeights(MLPModel model) {
        // We access package‑private fields via getters if available,
        // but since we are in the same package we can use the fields directly.
        // If those fields are private, MLPModel must expose getters like getW1() etc.
        // For safety, we assume the model provides a method getWeights() or we can
        // replicate the known array sizes.
        // As SequentialBaseline is in the baseline package and MLPModel is in model,
        // we may need public getters. Let's assume MLPModel has package-private fields
        // and is in com.distributed.mlp.model, so we need to add a public method
        // in MLPModel to expose the weight arrays, or we can hard-code the sizes
        // and copy them out via reflection? Better to add a simple getter.
        //
        // For this answer, we'll show the correct flattening using the known dimensions,
        // and we assume we added a temporary method to MLPModel:
        // public double[] getWeightsFlat() { ... }
        //
        // If you cannot modify MLPModel, you can use reflection or copy from known
        // public fields if they exist. We'll provide a version that uses reflection
        // to remain self-contained.

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

            int total =        MLPModel.INPUT_DIM  * MLPModel.HIDDEN1_DIM
                    +          MLPModel.HIDDEN1_DIM
                    + MLPModel.HIDDEN1_DIM * MLPModel.HIDDEN2_DIM
                    +          MLPModel.HIDDEN2_DIM
                    + MLPModel.HIDDEN2_DIM * MLPModel.OUTPUT_DIM
                    +          MLPModel.OUTPUT_DIM;

            double[] flat = new double[total];
            int idx = 0;

            // Row‑major copy
            for (double[] row : W1) for (double v : row) flat[idx++] = v;
            for (double v : B1) flat[idx++] = v;
            for (double[] row : W2) for (double v : row) flat[idx++] = v;
            for (double v : B2) flat[idx++] = v;
            for (double[] row : W3) for (double v : row) flat[idx++] = v;
            for (double v : B3) flat[idx++] = v;

            return flat;

        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException("Cannot access model weights – please ensure MLPModel fields are accessible.", e);
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