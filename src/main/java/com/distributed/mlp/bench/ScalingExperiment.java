package com.distributed.mlp.bench;

import com.distributed.mlp.baseline.SequentialBaseline;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * Runs strong and weak scaling experiments for distributed training and writes
 * analysis-ready CSV outputs.
 */
public final class ScalingExperiment {
    private static final int[] WORKER_CONFIGS = {1, 2, 3, 4, 8};
    private static final int EPOCHS = 5;
    private static final int STRONG_INPUT_SIZE = 75_750;
    private static final int WEAK_WORK_PER_WORKER = 25_250;

    private static final int MASTER_PORT = 9000;
    private static final int MASTER_TARGET_UPDATES_FALLBACK = 100;
    private static final int MINI_BATCH_SIZE = 32;
    private static final long BASE_SEED = 42L;

    private static final Path RESULTS_DIR = Path.of("results");
    private static final Path STRONG_CSV = RESULTS_DIR.resolve("strong_scaling.csv");
    private static final Path WEAK_CSV = RESULTS_DIR.resolve("weak_scaling.csv");
    private static final Path SEQ_CSV = RESULTS_DIR.resolve("sequential_results.csv");

    private ScalingExperiment() {
    }

    public static void main(String[] args) {
        try {
            run();
            System.out.println("Scaling experiments complete.");
            System.out.println("Wrote: " + STRONG_CSV.toAbsolutePath());
            System.out.println("Wrote: " + WEAK_CSV.toAbsolutePath());
        } catch (Exception e) {
            System.err.println("ScalingExperiment failed: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    public static void run() throws Exception {
        Files.createDirectories(RESULTS_DIR);

        List<ScalingRow> strongRows = runStrongScaling();
        List<ScalingRow> weakRows = runWeakScaling();

        writeCsv(STRONG_CSV, strongRows);
        writeCsv(WEAK_CSV, weakRows);
    }

    private static List<ScalingRow> runStrongScaling() throws Exception {
        List<RawEpoch> raw = new ArrayList<>();
        for (int workers : WORKER_CONFIGS) {
            raw.addAll(runConfig("strong", workers, STRONG_INPUT_SIZE));
        }

        double tSeq = meanWall(raw, "strong", 1);
        if (Double.isNaN(tSeq) || tSeq <= 0.0) {
            throw new IOException("Unable to compute strong scaling baseline (p=1)");
        }

        return toScaledRows(raw, tSeq);
    }

    private static List<ScalingRow> runWeakScaling() throws Exception {
        List<RawEpoch> raw = new ArrayList<>();
        for (int workers : WORKER_CONFIGS) {
            int inputSize = workers * WEAK_WORK_PER_WORKER;
            raw.addAll(runConfig("weak", workers, inputSize));
        }

        double tSeq = meanWall(raw, "weak", 1);
        if (Double.isNaN(tSeq) || tSeq <= 0.0) {
            throw new IOException("Unable to compute weak scaling baseline (p=1)");
        }

        return toScaledRows(raw, tSeq);
    }

    private static List<RawEpoch> runConfig(String experiment, int workers, int inputSize)
            throws IOException, InterruptedException {
        if (workers == 1) {
            return runSequential(experiment, inputSize);
        }
        return runAsyncSgd(experiment, workers, inputSize);
    }

    private static List<RawEpoch> runSequential(String experiment, int inputSize) throws IOException {
        List<RawEpoch> out = new ArrayList<>();
        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            long seed = BASE_SEED + inputSize + epoch;
            SequentialBaseline.run(1, seed);

            List<String> lines = Files.readAllLines(SEQ_CSV, StandardCharsets.UTF_8);
            if (lines.size() < 2) {
                throw new IOException("Sequential baseline CSV missing epoch output");
            }
            String[] cols = lines.get(1).split(",");
            if (cols.length < 4) {
                throw new IOException("Invalid sequential CSV row: " + lines.get(1));
            }

            double wallSec = Double.parseDouble(cols[3].trim());
            out.add(new RawEpoch(experiment, 1, inputSize, epoch, wallSec));

            System.out.printf(
                    Locale.ROOT,
                    "[ScalingExperiment] %s p=1 input=%d epoch=%d wall_sec=%.3f%n",
                    experiment,
                    inputSize,
                    epoch,
                    wallSec);
        }
        return out;
    }

    private static List<RawEpoch> runAsyncSgd(String experiment, int workers, int inputSize)
            throws IOException, InterruptedException {
        List<RawEpoch> rows = new ArrayList<>();
        int stepsPerWorkerPerEpoch = Math.max(1, inputSize / (workers * MINI_BATCH_SIZE));
        int minStepsForShutdown = (int) Math.ceil((double) MASTER_TARGET_UPDATES_FALLBACK / workers);
        int steps = Math.max(stepsPerWorkerPerEpoch, minStepsForShutdown);

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            Process master = startJavaProcess(List.of(
                    "com.distributed.mlp.Master",
                    String.valueOf(MASTER_PORT),
                    String.valueOf(workers)));

            Thread.sleep(1500L);

            List<Process> workerProcesses = new ArrayList<>();
            for (int workerId = 0; workerId < workers; workerId++) {
                Process worker = startJavaProcess(List.of(
                        "com.distributed.mlp.Worker",
                        "127.0.0.1",
                        String.valueOf(MASTER_PORT),
                        String.valueOf(workerId),
                        String.valueOf(workers),
                        String.valueOf(steps),
                        String.valueOf(BASE_SEED + inputSize + epoch + workerId)));
                workerProcesses.add(worker);
            }

            Instant t0 = Instant.now();
            int masterExit = master.waitFor();
            double wallSec = Duration.between(t0, Instant.now()).toMillis() / 1000.0;

            for (Process worker : workerProcesses) {
                worker.waitFor();
            }

            if (masterExit != 0) {
                destroyAll(workerProcesses);
                throw new IOException("Master exited with code " + masterExit + " for p=" + workers);
            }

            rows.add(new RawEpoch(experiment, workers, inputSize, epoch, wallSec));
            System.out.printf(
                    Locale.ROOT,
                    "[ScalingExperiment] %s p=%d input=%d epoch=%d wall_sec=%.3f%n",
                    experiment,
                    workers,
                    inputSize,
                    epoch,
                    wallSec);
        }

        return rows;
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
            throw new IllegalStateException("target/classes not found. Run 'mvn package' first.");
        }
        return classes.toString();
    }

    private static void destroyAll(List<Process> processes) {
        for (Process p : processes) {
            if (p.isAlive()) {
                p.destroyForcibly();
            }
        }
    }

    private static double meanWall(List<RawEpoch> rows, String experiment, int workers) {
        return rows.stream()
                .filter(r -> experiment.equals(r.experiment()))
                .filter(r -> r.workers() == workers)
                .mapToDouble(RawEpoch::wallSec)
                .average()
                .orElse(Double.NaN);
    }

    private static List<ScalingRow> toScaledRows(List<RawEpoch> raw, double tSeq) {
        List<ScalingRow> out = new ArrayList<>();
        for (RawEpoch row : raw) {
            int p = row.workers();
            double speedup = tSeq / row.wallSec();
            double efficiency = speedup / p;
            double parallelFraction = p == 1
                    ? Double.NaN
                    : ((1.0 / speedup) - 1.0) / ((1.0 / p) - 1.0);

            out.add(new ScalingRow(
                    row.experiment(),
                    p,
                    row.inputSize(),
                    row.epoch(),
                    row.wallSec(),
                    speedup,
                    efficiency,
                    parallelFraction));
        }

        return out.stream()
                .sorted(Comparator
                        .comparing(ScalingRow::experiment)
                        .thenComparingInt(ScalingRow::workers)
                        .thenComparingInt(ScalingRow::epoch))
                .toList();
    }

    private static void writeCsv(Path path, List<ScalingRow> rows) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("experiment,workers,input_size,epoch,wall_sec,speedup,efficiency,parallel_fraction")
                .append(System.lineSeparator());

        for (ScalingRow row : rows) {
            sb.append(String.format(
                    Locale.ROOT,
                    "%s,%d,%d,%d,%.6f,%.6f,%.6f,%s%n",
                    row.experiment(),
                    row.workers(),
                    row.inputSize(),
                    row.epoch(),
                    row.wallSec(),
                    row.speedup(),
                    row.efficiency(),
                    Double.isNaN(row.parallelFraction())
                            ? "NaN"
                            : String.format(Locale.ROOT, "%.6f", row.parallelFraction())));
        }

        Files.writeString(
                path,
                sb.toString(),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private record RawEpoch(String experiment, int workers, int inputSize, int epoch, double wallSec) {
    }

    private record ScalingRow(
            String experiment,
            int workers,
            int inputSize,
            int epoch,
            double wallSec,
            double speedup,
            double efficiency,
            double parallelFraction) {
    }
}
