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

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.model.MathUtils;
import com.distributed.mlp.protocol.WeightSerializer;

/**
 * Single‑threaded baseline that iterates a full dataset shard (worker 0 of 1)
 * and writes per‑epoch timing/metrics to CSV.
 * <p>
 * After training, the model is saved to {@code results/sequential_model.bin}.
 */
public final class SequentialBaseline {
    private static final int DEFAULT_EPOCHS = 3;
    private static final long DEFAULT_SEED = 42L;
    private static final double LEARNING_RATE = 1e-3;
    private static final Path OUTPUT_CSV = Path.of("results", "sequential_results.csv");
    private static final Path MODEL_PATH = Path.of("results", "sequential_model.bin");

    private SequentialBaseline() {
    }

    public static void main(String[] args) {
        int epochs = args.length >= 1 ? Integer.parseInt(args[0]) : DEFAULT_EPOCHS;
        long seed = args.length >= 2 ? Long.parseLong(args[1]) : DEFAULT_SEED;
        int inputSize = args.length >= 3 ? Integer.parseInt(args[2]) : 0;

        try {
            run(epochs, seed, inputSize);
        } catch (IOException e) {
            System.err.println("SequentialBaseline failed: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    public static void run(int epochs, long seed, int inputSize) throws IOException {
        if (epochs <= 0) {
            throw new IllegalArgumentException("epochs must be > 0");
        }

        System.out.println("[SequentialBaseline] Loading dataset...");
        DataLoader dataLoader = new DataLoader();
        List<Sample> dataset = dataLoader.loadShard(0, 1);
        if (dataset.isEmpty()) {
            throw new IOException("Dataset is empty. Check data/cifar-10-batches-bin/ path.");
        }
        if (inputSize > 0 && dataset.size() > inputSize) {
            dataset = dataset.subList(0, inputSize);
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
                List<Sample> epochSamples = new ArrayList<>(dataset);
                Collections.shuffle(epochSamples, new Random(seed + epoch));

                for (Sample sample : epochSamples) {
                    double[] probs = model.forward(sample.pixels());
                    int prediction = argmax(probs);
                    if (prediction == sample.label()) {
                        correct++;
                    }

                    totalLoss += MathUtils.crossEntropyLoss(probs, sample.label());

                    MLPModel.Gradient gradient = model.backward(sample.pixels(), sample.label());
                    applyGradient(model, gradient, LEARNING_RATE);
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