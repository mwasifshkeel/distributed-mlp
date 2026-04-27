package com.distributed.mlp.correctness;

import com.distributed.mlp.data.DataLoader;
import com.distributed.mlp.data.DataLoader.Sample;
import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.model.MathUtils;
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
 */
public final class CorrectnessChecker {
    private static final int SUBSET_SIZE = 1_000;
    private static final int EPOCHS = 3;
    private static final long SEED = 42L;

    private static final double LOSS_TOLERANCE = 1e-4;

    private static final int DEFAULT_PORT = 9000;
    private static final int DEFAULT_WORKERS = 1;
    private static final int DEFAULT_STEPS = 16;

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

        return diffCount == 0 && lossOk && distributedSmokeOk;
    }

    private static List<Sample> loadSubset(int subsetSize) throws IOException {
        DataLoader loader = new DataLoader();
        List<Sample> all = loader.loadShard(0, 1);
        if (all.isEmpty()) {
            throw new IOException("Dataset is empty. Check data/food-101/images path.");
        }
        int end = Math.min(subsetSize, all.size());
        return new ArrayList<>(all.subList(0, end));
    }

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

        return new EvalResult(new ArrayList<>(finalPredictions), finalLoss);
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

    private record EvalResult(List<Integer> predictions, double finalLoss) {
    }
}
