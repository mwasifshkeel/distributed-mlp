package com.distributed.mlp.bench;

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
 * Runs baseline vs optimized distributed training and writes a comparison CSV.
 */
public final class OptimizationBenchmark {
    private static final int DEFAULT_PORT = 9000;
    private static final int DEFAULT_WORKERS = 3;
    private static final int DEFAULT_EPOCHS = 3;
    private static final int DEFAULT_PULL_EVERY = 10;
    private static final long BASE_SEED = 42L;

    private static final int TOTAL_SAMPLES = 5_000;
    private static final int MINI_BATCH = 32;

    private static final Path RESULTS_DIR = Path.of("results");
    private static final Path OPT_CSV = RESULTS_DIR.resolve("optimisation_runs.csv");

    private OptimizationBenchmark() {
    }

    public static void main(String[] args) {
        int port = args.length >= 1 ? Integer.parseInt(args[0]) : DEFAULT_PORT;
        int workers = args.length >= 2 ? Integer.parseInt(args[1]) : DEFAULT_WORKERS;
        int epochs = args.length >= 3 ? Integer.parseInt(args[2]) : DEFAULT_EPOCHS;
        int pullEvery = args.length >= 4 ? Integer.parseInt(args[3]) : DEFAULT_PULL_EVERY;

        try {
            run(port, workers, epochs, pullEvery);
        } catch (Exception e) {
            System.err.println("OptimizationBenchmark failed: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    public static void run(int port, int workers, int epochs, int pullEvery)
            throws IOException, InterruptedException {
        if (workers <= 0 || epochs <= 0) {
            throw new IllegalArgumentException("workers and epochs must be > 0");
        }
        if (pullEvery <= 0) {
            throw new IllegalArgumentException("pullEvery must be > 0");
        }

        Files.createDirectories(RESULTS_DIR);
        initCsv();

        int stepsPerWorker = (TOTAL_SAMPLES / workers / MINI_BATCH) * epochs;
        if (stepsPerWorker <= 0) {
            throw new IllegalArgumentException("computed stepsPerWorker is 0; check workers/epochs");
        }

        RunResult baseline = runDistributed(
            "baseline",
            port,
            workers,
            stepsPerWorker,
            BASE_SEED,
            false,
            1);
        appendRow(baseline);

        RunResult optimized = runDistributed(
            "optimized",
            port,
            workers,
            stepsPerWorker,
            BASE_SEED,
            true,
            pullEvery);
        appendRow(optimized);

        double speedup = baseline.wallSec() / optimized.wallSec();
        System.out.printf(
            Locale.ROOT,
            "[OptimizationBenchmark] baseline=%.3fs optimized=%.3fs speedup=%.3fx%n",
            baseline.wallSec(),
            optimized.wallSec(),
            speedup);
        System.out.println("Wrote: " + OPT_CSV.toAbsolutePath());
    }

    private static RunResult runDistributed(
            String mode,
            int port,
            int workers,
            int stepsPerWorker,
            long seed,
            boolean compress,
            int pullEvery) throws IOException, InterruptedException {

        List<String> masterCmd = new ArrayList<>();
        masterCmd.add(getJavaExecutable());
        masterCmd.addAll(javaProps(compress, pullEvery));
        masterCmd.add("-Xmx256m");
        masterCmd.add("-cp");
        masterCmd.add(resolveRuntimeClasspath());
        masterCmd.addAll(List.of(
                "com.distributed.mlp.Master",
                String.valueOf(port),
                String.valueOf(workers),
                String.valueOf(stepsPerWorker),
                String.valueOf(seed),
                "false"));

        Process master = null;
        List<Process> workersProcs = new ArrayList<>();
        try {
            master = new ProcessBuilder(masterCmd)
                    .redirectErrorStream(true)
                    .inheritIO()
                    .start();

            Thread.sleep(1500L);

            for (int workerId = 0; workerId < workers; workerId++) {
                List<String> workerCmd = new ArrayList<>();
                workerCmd.add(getJavaExecutable());
                workerCmd.addAll(javaProps(compress, pullEvery));
                workerCmd.add("-Xmx512m");
                workerCmd.add("-cp");
                workerCmd.add(resolveRuntimeClasspath());
                workerCmd.addAll(List.of(
                        "com.distributed.mlp.Worker",
                        "127.0.0.1",
                        String.valueOf(port),
                        String.valueOf(workerId),
                        String.valueOf(workers),
                        String.valueOf(stepsPerWorker),
                        String.valueOf(seed + workerId)));
                workersProcs.add(new ProcessBuilder(workerCmd)
                        .redirectErrorStream(true)
                        .inheritIO()
                        .start());
            }

            Instant t0 = Instant.now();
            int masterExit = master.waitFor();
            double wallSec = Duration.between(t0, Instant.now()).toMillis() / 1000.0;

            for (Process worker : workersProcs) {
                worker.waitFor();
            }

            if (masterExit != 0) {
                throw new IOException("Master exited with code " + masterExit + " for mode=" + mode);
            }

            return new RunResult(mode, workers, stepsPerWorker, pullEvery, compress, wallSec);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw e;
        } finally {
            destroyAll(workersProcs);
            if (master != null && master.isAlive()) {
                master.destroyForcibly();
            }
        }
    }

    private static List<String> javaProps(boolean compress, int pullEvery) {
        List<String> props = new ArrayList<>();
        if (compress) {
            props.add("-Dmlp.compressGradients=true");
        }
        if (pullEvery > 1) {
            props.add("-Dmlp.pullEvery=" + pullEvery);
        }
        return props;
    }

    private static void initCsv() throws IOException {
        String header = "mode,workers,steps_per_worker,pull_every,compress,wall_sec" + System.lineSeparator();
        Files.writeString(
                OPT_CSV,
                header,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static void appendRow(RunResult result) throws IOException {
        String line = String.format(
                Locale.ROOT,
                "%s,%d,%d,%d,%s,%.6f%n",
                result.mode(),
                result.workers(),
                result.stepsPerWorker(),
                result.pullEvery(),
                result.compress(),
                result.wallSec());
        Files.writeString(
                OPT_CSV,
                line,
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND);
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

    private static void destroyAll(List<Process> processes) {
        for (Process p : processes) {
            if (p.isAlive()) {
                p.destroyForcibly();
            }
        }
    }

    private record RunResult(
            String mode,
            int workers,
            int stepsPerWorker,
            int pullEvery,
            boolean compress,
            double wallSec) {
    }
}
