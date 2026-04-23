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
import java.util.List;
import java.util.Locale;

/**
 * Orchestrates benchmark runs for sequential and async distributed modes and
 * writes a normalized CSV table for later scaling/speedup analysis.
 */
public final class BenchmarkRunner {
    private static final int[] WORKER_CONFIGS = {1, 2, 4, 8};
    private static final int[] INPUT_SIZES = {10_000, 40_000, 75_750};
    private static final int EPOCHS = 5;
    private static final long BASE_SEED = 42L;
    private static final int MASTER_PORT = 9000;
    private static final int MASTER_TARGET_UPDATES_FALLBACK = 100;
    private static final int MINI_BATCH_SIZE = 32;

    private static final Path RESULTS_DIR = Path.of("results");
    private static final Path RAW_CSV = RESULTS_DIR.resolve("raw.csv");
    private static final Path SEQ_CSV = RESULTS_DIR.resolve("sequential_results.csv");

    private BenchmarkRunner() {
    }

    public static void main(String[] args) {
        try {
            runBenchmarks();
            System.out.println("Benchmark run complete. CSV written: " + RAW_CSV.toAbsolutePath());
        } catch (Exception e) {
            System.err.println("BenchmarkRunner failed: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    /**
     * Runs all benchmark configurations and writes rows in the expected schema:
     * mode,workers,threads,input_size,epoch,wall_sec,loss,accuracy
     */
    public static void runBenchmarks() throws Exception {
        Files.createDirectories(RESULTS_DIR);
        initRawCsv();

        for (int inputSize : INPUT_SIZES) {
            List<EpochResult> seqEpochs = runSequential(inputSize, EPOCHS);
            appendRows("sequential", 1, 1, inputSize, seqEpochs);

            for (int workers : WORKER_CONFIGS) {
                if (workers == 1) {
                    continue;
                }
                int threadsPerWorker = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
                List<EpochResult> asyncEpochs = runAsyncSgd(workers, inputSize, EPOCHS, BASE_SEED + inputSize);
                appendRows("async_sgd", workers, threadsPerWorker, inputSize, asyncEpochs);
            }
        }
    }

    private static List<EpochResult> runSequential(int inputSize, int epochs) throws IOException {
        long seed = BASE_SEED + inputSize;
        SequentialBaseline.run(epochs, seed);

        List<String> lines = Files.readAllLines(SEQ_CSV, StandardCharsets.UTF_8);
        List<EpochResult> out = new ArrayList<>();

        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }
            String[] cols = line.split(",");
            if (cols.length < 4) {
                throw new IOException("Invalid sequential CSV row: " + line);
            }
            int epoch = Integer.parseInt(cols[0].trim());
            double loss = Double.parseDouble(cols[1].trim());
            double accuracy = Double.parseDouble(cols[2].trim());
            double wallSec = Double.parseDouble(cols[3].trim());
            out.add(new EpochResult(epoch, wallSec, loss, accuracy));
        }

        if (out.size() != epochs) {
            throw new IOException("Expected " + epochs + " sequential epochs but found " + out.size());
        }
        return out;
    }

    private static List<EpochResult> runAsyncSgd(int workers, int inputSize, int epochs, long seed)
            throws IOException, InterruptedException {
        List<EpochResult> rows = new ArrayList<>();

        int stepsPerWorkerPerEpoch = Math.max(1, inputSize / (workers * MINI_BATCH_SIZE));
        int minStepsNeededForShutdown = (int) Math.ceil((double) MASTER_TARGET_UPDATES_FALLBACK / workers);
        int steps = Math.max(stepsPerWorkerPerEpoch, minStepsNeededForShutdown);

        for (int epoch = 1; epoch <= epochs; epoch++) {
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
                        String.valueOf(seed + epoch + workerId)));
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
                throw new IOException("Master exited with non-zero code " + masterExit + " for workers=" + workers);
            }

            rows.add(new EpochResult(epoch, wallSec, Double.NaN, Double.NaN));
            System.out.printf(
                    Locale.ROOT,
                    "[BenchmarkRunner] mode=async_sgd workers=%d input_size=%d epoch=%d wall_sec=%.3f%n",
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
            throw new IllegalStateException("target/classes not found. Run 'mvn package' before BenchmarkRunner.");
        }
        return classes.toString();
    }

    private static void initRawCsv() throws IOException {
        String header = "mode,workers,threads,input_size,epoch,wall_sec,loss,accuracy" + System.lineSeparator();
        Files.writeString(
                RAW_CSV,
                header,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static void appendRows(String mode, int workers, int threads, int inputSize, List<EpochResult> rows)
            throws IOException {
        StringBuilder sb = new StringBuilder();
        for (EpochResult row : rows) {
            sb.append(String.format(
                    Locale.ROOT,
                    "%s,%d,%d,%d,%d,%.6f,%s,%s%n",
                    mode,
                    workers,
                    threads,
                    inputSize,
                    row.epoch(),
                    row.wallSec(),
                    formatMaybeNan(row.loss()),
                    formatMaybeNan(row.accuracy())));
        }

        Files.writeString(
                RAW_CSV,
                sb.toString(),
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
    }

    private static String formatMaybeNan(double value) {
        if (Double.isNaN(value)) {
            return "NaN";
        }
        return String.format(Locale.ROOT, "%.6f", value);
    }

    private static void destroyAll(List<Process> processes) {
        for (Process p : processes) {
            if (p.isAlive()) {
                p.destroyForcibly();
            }
        }
    }

    private record EpochResult(int epoch, double wallSec, double loss, double accuracy) {
    }
}
