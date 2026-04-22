package com.distributed.mlp.baseline;

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.model.MathUtils;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Locale;

/**
 * Single-threaded baseline that iterates a full dataset shard (worker 0 of 1)
 * and writes per-epoch timing/metrics to CSV.
 */
public final class SequentialBaseline {
    private static final int DEFAULT_EPOCHS = 5;
    private static final long DEFAULT_SEED = 42L;
    private static final Path OUTPUT_CSV = Path.of("results", "sequential_results.csv");

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

        DataLoader dataLoader = new DataLoader();
        List<Sample> dataset = dataLoader.loadShard(0, 1);
        if (dataset.isEmpty()) {
            throw new IOException("Dataset is empty. Check data/food-101/images path.");
        }

        MLPModel model = new MLPModel();
        model.initXavier(seed);

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

                    // Backward pass is kept to match compute flow with distributed workers.
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

        System.out.println("Sequential baseline CSV written: " + OUTPUT_CSV.toString());
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
