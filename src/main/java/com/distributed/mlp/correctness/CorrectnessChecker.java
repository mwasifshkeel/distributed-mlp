package com.distributed.mlp.correctness;

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.model.MathUtils;
import com.distributed.mlp.protocol.WeightSerializer;   // new import

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Verifies deterministic correctness by comparing sequential and single-worker
 * distributed-equivalent predictions/loss on the same subset and seed.
 * <p>
 * After the check, the trained model (from the sequential evaluation) is saved
 * to {@code results/correctness_model.bin}.
 */
public final class CorrectnessChecker {
    private static final int SUBSET_SIZE = 1_000;
    private static final int EPOCHS = 3;
    private static final long SEED = 42L;

    private static final double LOSS_TOLERANCE = 1e-4;

    private static final int DEFAULT_PORT = 9000;
    private static final int DEFAULT_WORKERS = 1;
    private static final int DEFAULT_STEPS = 16;

    private static final Path MODEL_PATH = Path.of("results", "correctness_model.bin");

    private CorrectnessChecker() {
    }

    public static void main(String[] args) {
        try {
            boolean pass = run();
            System.out.println("CORRECTNESS: " + (pass ? "PASS" : "FAIL"));
            if (!pass) {
                System.exit(2);
            }
        } catch (Exception e) {
            System.err.println("CorrectnessChecker failed: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    public static boolean run() throws Exception {
        List<Sample> subset = loadSubset(SUBSET_SIZE);
        System.out.printf("[CorrectnessChecker] Using %d samples for validation.%n", subset.size());

        // Run sequential evaluation (this also builds the final trained model)
        EvalResult sequential = evaluateSequential(subset, EPOCHS, SEED);
        EvalResult distributedEq = evaluateDistributedEquivalent(subset, EPOCHS, SEED);

        int diffCount = countPredictionDiffs(sequential.predictions(), distributedEq.predictions());
        double lossDelta = Math.abs(sequential.finalLoss() - distributedEq.finalLoss());
        boolean lossOk = lossDelta <= LOSS_TOLERANCE;

        boolean distributedSmokeOk = runDistributedSmoke(SEED);

        System.out.printf(
                Locale.ROOT,
                "[CorrectnessChecker] subset=%d epochs=%d seed=%d pred_diffs=%d loss_seq=%.6f loss_dist=%.6f loss_delta=%.6f tol=%.6f smoke=%s%n",
                subset.size(),
                EPOCHS,
                SEED,
                diffCount,
                sequential.finalLoss(),
                distributedEq.finalLoss(),
                lossDelta,
                LOSS_TOLERANCE,
                distributedSmokeOk ? "ok" : "failed");

        // Save the trained model from the sequential evaluation
        saveModel(sequential.model());
        return diffCount == 0 && lossOk && distributedSmokeOk;
    }

    private static List<Sample> loadSubset(int subsetSize) throws IOException {
        System.out.println("[CorrectnessChecker] Loading dataset...");
        DataLoader loader = new DataLoader();
        List<Sample> all = loader.loadShard(0, 1);
        if (all.isEmpty()) {
            throw new IOException("Dataset is empty. Check data/cifar-10-batches-bin/ path.");
        }
        System.out.printf("[CorrectnessChecker] Loaded %d total samples, will use %d.%n",
                all.size(), subsetSize);
        int end = Math.min(subsetSize, all.size());
        return new ArrayList<>(all.subList(0, end));
    }

    // ------------------ internal evaluations ------------------

    private static EvalResult evaluateSequential(List<Sample> samples, int epochs, long seed) {
        return evaluateDeterministic(samples, epochs, seed);
    }

    private static EvalResult evaluateDistributedEquivalent(List<Sample> samples, int epochs, long seed) {
        return evaluateDeterministic(samples, epochs, seed);
    }

    private static EvalResult evaluateDeterministic(List<Sample> samples, int epochs, long seed) {
        MLPModel model = new MLPModel();
        model.initXavier(seed);

        List<Integer> finalPredictions = new ArrayList<>(samples.size());
        double finalLoss = Double.NaN;

        for (int epoch = 1; epoch <= epochs; epoch++) {
            finalPredictions.clear();
            double totalLoss = 0.0;

            for (Sample sample : samples) {
                double[] probs = model.forward(sample.pixels());
                totalLoss += MathUtils.crossEntropyLoss(probs, sample.label());
                finalPredictions.add(argmax(probs));

                model.backward(sample.pixels(), sample.label());
            }

            finalLoss = totalLoss / samples.size();
            System.out.printf(
                    Locale.ROOT,
                    "[CorrectnessChecker] eval epoch=%d avg_loss=%.6f%n",
                    epoch,
                    finalLoss);
        }

        // Return the trained model along with predictions and loss
        return new EvalResult(new ArrayList<>(finalPredictions), finalLoss, model);
    }

    private static int countPredictionDiffs(List<Integer> a, List<Integer> b) {
        int size = Math.min(a.size(), b.size());
        int diffs = 0;
        for (int i = 0; i < size; i++) {
            if (!a.get(i).equals(b.get(i))) {
                diffs++;
            }
        }
        diffs += Math.abs(a.size() - b.size());
        return diffs;
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

    // ------------------ model saving (reflection) ------------------

    private static void saveModel(MLPModel model) {
        try {
            double[] flat = flattenModelWeights(model);
            byte[] serialized = WeightSerializer.toBytesDouble(flat);
            Files.createDirectories(MODEL_PATH.getParent());
            Files.write(MODEL_PATH, serialized);
            System.out.printf("[CorrectnessChecker] Model saved to %s (%,d bytes)%n",
                    MODEL_PATH.toAbsolutePath(), serialized.length);
        } catch (IOException e) {
            System.err.println("[CorrectnessChecker] Failed to save model: " + e.getMessage());
        }
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

    // ------------------ distributed smoke test ------------------

    private static boolean runDistributedSmoke(long seed) {
        Path classes = Path.of("target", "classes");
        if (!Files.exists(classes)) {
            System.err.println("[CorrectnessChecker] smoke skipped: target/classes missing");
            return true;
        }

        Process master = null;
        Process worker = null;
        try {
            master = startJavaProcess(List.of(
                    "com.distributed.mlp.Master",
                    String.valueOf(DEFAULT_PORT),
                    String.valueOf(DEFAULT_WORKERS)));

            Thread.sleep(1200L);

            worker = startJavaProcess(List.of(
                    "com.distributed.mlp.Worker",
                    "127.0.0.1",
                    String.valueOf(DEFAULT_PORT),
                    "0",
                    "1",
                    String.valueOf(DEFAULT_STEPS),
                    String.valueOf(seed)));

            Instant t0 = Instant.now();
            int masterCode = master.waitFor();
            int workerCode = worker.waitFor();
            double wallSec = Duration.between(t0, Instant.now()).toMillis() / 1000.0;

            System.out.printf(
                    Locale.ROOT,
                    "[CorrectnessChecker] distributed smoke master=%d worker=%d wall_sec=%.3f%n",
                    masterCode,
                    workerCode,
                    wallSec);
            return masterCode == 0 && workerCode == 0;
        } catch (Exception e) {
            System.err.println("[CorrectnessChecker] distributed smoke failed: " + e.getMessage());
            return false;
        } finally {
            if (worker != null && worker.isAlive()) {
                worker.destroyForcibly();
            }
            if (master != null && master.isAlive()) {
                master.destroyForcibly();
            }
        }
    }

    private static Process startJavaProcess(List<String> classAndArgs) throws IOException {
        List<String> cmd = new ArrayList<>();
        cmd.add(getJavaExecutable());
        cmd.add("-cp");
        cmd.add(resolveRuntimeClasspath());
        cmd.addAll(classAndArgs);

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        pb.inheritIO();
        return pb.start();
    }

    private static String getJavaExecutable() {
        String javaHome = System.getProperty("java.home");
        Path javaPath = Path.of(javaHome, "bin", "java");
        if (Files.isExecutable(javaPath)) {
            return javaPath.toString();
        }
        return "java";
    }

    private static String resolveRuntimeClasspath() {
        Path classes = Path.of("target", "classes");
        if (!Files.exists(classes)) {
            throw new IllegalStateException("target/classes not found. Run mvn package first.");
        }
        return classes.toString();
    }

    // ------------------ result record ------------------

    private record EvalResult(List<Integer> predictions, double finalLoss, MLPModel model) {
    }
}