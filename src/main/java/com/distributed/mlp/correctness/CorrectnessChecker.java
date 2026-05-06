package com.distributed.mlp.correctness;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.model.MathUtils;
import com.distributed.mlp.protocol.WeightSerializer;

/**
 * Verifies deterministic correctness by comparing sequential-equivalent training
 * with a single-worker distributed run on the same inputs and seed.
 * <p>
 * After the check, the trained model (from the sequential evaluation) is saved
 * to {@code results/correctness_model.bin}.
 */
public final class CorrectnessChecker {
    private static final int MINI_BATCH_SIZE = 32;
    private static final int DEFAULT_STEPS = 8;
    private static final long SEED = 42L;

    private static final double LEARNING_RATE = 1e-3;
    private static final double LOSS_TOLERANCE = 1e-9;
    private static final double WEIGHT_TOLERANCE = 1e-9;

    private static final int DEFAULT_PORT = 9000;
    private static final int DEFAULT_WORKERS = 1;
    private static final Duration SMOKE_TIMEOUT = Duration.ofSeconds(90);

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
        int steps = DEFAULT_STEPS;
        int sampleCount = steps * MINI_BATCH_SIZE;
        List<Sample> subset = loadSubset(sampleCount);
        System.out.printf("[CorrectnessChecker] Using %d samples for validation.%n", subset.size());

        EvalResult sequential = trainSequential(subset, steps);
        DistributedResult distributed = runDistributedAndLoadModel(subset, steps, SEED);

        if (!distributed.ok()) {
            saveModel(sequential.model());
            return false;
        }

        EvalResult distributedEval = distributed.eval();

        int diffCount = countPredictionDiffs(sequential.predictions(), distributedEval.predictions());
        double lossDelta = Math.abs(sequential.finalLoss() - distributedEval.finalLoss());
        boolean lossOk = lossDelta <= LOSS_TOLERANCE;

        double maxWeightDelta = maxAbsDiff(
            flattenModelWeights(sequential.model()),
            flattenModelWeights(distributedEval.model()));
        boolean weightsOk = maxWeightDelta <= WEIGHT_TOLERANCE;

        System.out.printf(
            Locale.ROOT,
            "[CorrectnessChecker] steps=%d seed=%d pred_diffs=%d loss_seq=%.9f loss_dist=%.9f loss_delta=%.9f tol=%.9f max_weight_delta=%.9f weight_tol=%.9f%n",
            steps,
            SEED,
            diffCount,
            sequential.finalLoss(),
            distributedEval.finalLoss(),
            lossDelta,
            LOSS_TOLERANCE,
            maxWeightDelta,
            WEIGHT_TOLERANCE);

        saveModel(sequential.model());
        return diffCount == 0 && lossOk && weightsOk;
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

    // ------------------ sequential-equivalent training ------------------

    private static EvalResult trainSequential(List<Sample> samples, int steps) {
        MLPModel model = new MLPModel();
        double[] zeros = new double[MLPModel.INPUT_DIM * MLPModel.HIDDEN1_DIM
            + MLPModel.HIDDEN1_DIM
            + MLPModel.HIDDEN1_DIM * MLPModel.HIDDEN2_DIM
            + MLPModel.HIDDEN2_DIM
            + MLPModel.HIDDEN2_DIM * MLPModel.OUTPUT_DIM
            + MLPModel.OUTPUT_DIM];
        model.loadWeights(zeros);

        int maxBatches = samples.size() / MINI_BATCH_SIZE;
        int effectiveSteps = Math.min(steps, maxBatches);
        if (effectiveSteps < steps) {
            System.out.printf("[CorrectnessChecker] Reducing steps from %d to %d (insufficient samples).%n",
                    steps, effectiveSteps);
        }

        for (int step = 0; step < effectiveSteps; step++) {
            int start = step * MINI_BATCH_SIZE;
            int end = start + MINI_BATCH_SIZE;
            List<Sample> miniBatch = samples.subList(start, end);

            double[] batchGrad = computeBatchGradient(model, miniBatch);
            double scale = 1.0 / miniBatch.size();
            for (int i = 0; i < batchGrad.length; i++) {
                batchGrad[i] *= scale;
            }
            applyGradient(model, batchGrad, LEARNING_RATE);
        }

        return evaluateModel(model, samples);
    }

    private static EvalResult evaluateModel(MLPModel model, List<Sample> samples) {
        List<Integer> finalPredictions = new ArrayList<>(samples.size());
        double totalLoss = 0.0;

        for (Sample sample : samples) {
            double[] probs = model.forward(sample.pixels());
            totalLoss += MathUtils.crossEntropyLoss(probs, sample.label());
            finalPredictions.add(argmax(probs));
        }

        double finalLoss = totalLoss / samples.size();
        return new EvalResult(new ArrayList<>(finalPredictions), finalLoss, model);
    }

    private static void applyGradient(MLPModel model, double[] gradient, double lr) {
        double[] flat = flattenModelWeights(model);
        for (int i = 0; i < flat.length; i++) {
            flat[i] -= lr * gradient[i];
        }
        model.loadWeights(flat);
    }

    private static double[] computeBatchGradient(MLPModel model, List<Sample> miniBatch) {
        double[] total = null;
        for (Sample sample : miniBatch) {
            MLPModel.Gradient grad = model.backward(sample.pixels(), sample.label());
            double[] flat = flattenGradient(grad);
            if (total == null) {
                total = new double[flat.length];
            }
            for (int i = 0; i < flat.length; i++) {
                total[i] += flat[i];
            }
        }
        if (total == null) {
            total = new double[MLPModel.INPUT_DIM * MLPModel.HIDDEN1_DIM
                    + MLPModel.HIDDEN1_DIM
                    + MLPModel.HIDDEN1_DIM * MLPModel.HIDDEN2_DIM
                    + MLPModel.HIDDEN2_DIM
                    + MLPModel.HIDDEN2_DIM * MLPModel.OUTPUT_DIM
                    + MLPModel.OUTPUT_DIM];
        }
        return total;
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

    private static double maxAbsDiff(double[] a, double[] b) {
        int size = Math.min(a.length, b.length);
        double max = 0.0;
        for (int i = 0; i < size; i++) {
            double diff = Math.abs(a[i] - b[i]);
            if (diff > max) max = diff;
        }
        return max;
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

    private static DistributedResult runDistributedAndLoadModel(List<Sample> subset, int steps, long seed) {
        Path classes = Path.of("target", "classes");
        if (!Files.exists(classes)) {
            System.err.println("[CorrectnessChecker] smoke skipped: target/classes missing");
            return new DistributedResult(false, null, null);
        }

        deleteExistingModelWeights();

        Process master = null;
        Process worker = null;
        try {
            master = startJavaProcess(List.of(
                "com.distributed.mlp.Master",
                String.valueOf(DEFAULT_PORT),
                String.valueOf(DEFAULT_WORKERS),
                String.valueOf(steps),
                String.valueOf(seed),
                "false"));

            Thread.sleep(1200L);

            worker = startJavaProcess(List.of(
                "com.distributed.mlp.Worker",
                "127.0.0.1",
                String.valueOf(DEFAULT_PORT),
                "0",
                "1",
                String.valueOf(steps),
                String.valueOf(seed)));

            Instant t0 = Instant.now();
            boolean masterDone = master.waitFor(SMOKE_TIMEOUT.toMillis(), TimeUnit.MILLISECONDS);
            boolean workerDone = worker.waitFor(SMOKE_TIMEOUT.toMillis(), TimeUnit.MILLISECONDS);
            int masterCode = masterDone ? master.exitValue() : -1;
            int workerCode = workerDone ? worker.exitValue() : -1;
            double wallSec = Duration.between(t0, Instant.now()).toMillis() / 1000.0;

            if (!masterDone) {
                System.err.println("[CorrectnessChecker] master timed out, destroying process");
                master.destroyForcibly();
            }
            if (!workerDone) {
                System.err.println("[CorrectnessChecker] worker timed out, destroying process");
                worker.destroyForcibly();
            }

            System.out.printf(
                    Locale.ROOT,
                    "[CorrectnessChecker] distributed smoke master=%d worker=%d wall_sec=%.3f%n",
                    masterCode,
                    workerCode,
                    wallSec);
            boolean ok = masterCode == 0 && workerCode == 0;
            if (!ok) {
                return new DistributedResult(false, null, null);
            }

            Path weightsPath = findLatestModelWeights();
            if (weightsPath == null) {
                System.err.println("[CorrectnessChecker] No model_weights_*.bin found after run.");
                return new DistributedResult(false, null, null);
            }

            double[] weights = WeightSerializer.fromBytesDouble(Files.readAllBytes(weightsPath));
            MLPModel model = new MLPModel();
            model.initXavier(seed);
            model.loadWeights(weights);

            EvalResult eval = evaluateModel(model, subset);
            return new DistributedResult(true, eval, weightsPath);
        } catch (IOException | RuntimeException e) {
            System.err.println("[CorrectnessChecker] distributed smoke failed: " + e.getMessage());
            return new DistributedResult(false, null, null);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("[CorrectnessChecker] distributed smoke interrupted: " + e.getMessage());
            return new DistributedResult(false, null, null);
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
        cmd.add("-Dmlp.computeThreads=1");
        cmd.add("-Dmlp.pullEvery=1");
        cmd.add("-Dmlp.compressGradients=false");
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

    private record DistributedResult(boolean ok, EvalResult eval, Path weightsPath) {
    }

    private static void deleteExistingModelWeights() {
        try {
            Path dir = Path.of("results");
            if (!Files.isDirectory(dir)) return;
            try (var stream = Files.newDirectoryStream(dir, "model_weights_*.bin")) {
                for (Path p : stream) {
                    Files.deleteIfExists(p);
                }
            }
        } catch (IOException e) {
            System.err.println("[CorrectnessChecker] Could not clear model weights: " + e.getMessage());
        }
    }

    private static Path findLatestModelWeights() throws IOException {
        Path dir = Path.of("results");
        if (!Files.isDirectory(dir)) return null;
        Path latest = null;
        long latestTs = -1L;
        try (var stream = Files.newDirectoryStream(dir, "model_weights_*.bin")) {
            for (Path p : stream) {
                long ts = Files.getLastModifiedTime(p).toMillis();
                if (ts > latestTs) {
                    latestTs = ts;
                    latest = p;
                }
            }
        }
        return latest;
    }

    private static double[] flattenGradient(MLPModel.Gradient gradient) {
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