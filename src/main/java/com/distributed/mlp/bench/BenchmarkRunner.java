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

    import com.distributed.mlp.baseline.SequentialBaseline;

    /**
     * Orchestrates benchmark runs for sequential and async distributed modes and
     * writes a normalized CSV table for later scaling/speedup analysis.
     */
    public final class BenchmarkRunner {
        private static final int[] WORKER_CONFIGS = {1, 2, 4, 6};
        private static final int[] INPUT_SIZES = {5_000, 10_000};
        private static final int EPOCHS = 1;
        private static final long BASE_SEED = 42L;
        private static final int MASTER_PORT = 9000;
        private static final int MASTER_TARGET_UPDATES_FALLBACK = 100;
        private static final int MINI_BATCH_SIZE = 128;
        private static final int STARTUP_GRACE_MS = 2_000;
        private static final int DEFAULT_PULL_EVERY = 10;

        private static final Path RESULTS_DIR = Path.of("results");
        private static final Path RAW_CSV = RESULTS_DIR.resolve("raw.csv");
        private static final Path SEQ_CSV = RESULTS_DIR.resolve("sequential_results.csv");
        private static final String MAX_SAMPLES_PROP = "mlp.maxSamples";

        private BenchmarkRunner() {
        }

        public static void main(String[] args) {
            try {
                configureMaxSamples(args);
                int limit = resolveMaxSamples();
                if (limit > 0) {
                    System.out.printf("[BenchmarkRunner] Sample cap enabled: %,d images%n", limit);
                }
                System.out.println("[BenchmarkRunner] Starting full benchmark suite...");
                runBenchmarks();
                System.out.println("Benchmark complete. CSV written: " + RAW_CSV.toAbsolutePath());
            } catch (Exception e) {
                System.err.println("BenchmarkRunner failed: " + e.getMessage());
                e.printStackTrace(System.err);
                System.exit(1);
            }
        }

        public static void runBenchmarks() throws Exception {
            Files.createDirectories(RESULTS_DIR);
            initRawCsv();

            for (int inputSize : INPUT_SIZES) {
                int effectiveInputSize = effectiveSampleCount(inputSize);
                System.out.printf("[BenchmarkRunner] Sequential baseline (input=%d, epochs=%d)%n", effectiveInputSize, EPOCHS);
                List<EpochResult> seqEpochs = runSequential(effectiveInputSize, EPOCHS);
                appendRows("sequential", 1, 1, effectiveInputSize, seqEpochs);

                for (int workers : WORKER_CONFIGS) {
                    if (workers == 1) continue;
                    System.out.printf("[BenchmarkRunner] Async SGD (workers=%d, input=%d, epochs=%d)%n", workers, effectiveInputSize, EPOCHS);
                    int threadsPerWorker = threadsPerWorker(workers);
                    List<EpochResult> asyncEpochs = runAsyncSgd(
                            workers,
                            threadsPerWorker,
                            effectiveInputSize,
                            EPOCHS,
                            BASE_SEED + effectiveInputSize);
                    appendRows("async_sgd", workers, threadsPerWorker, effectiveInputSize, asyncEpochs);
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

        private static List<EpochResult> runAsyncSgd(int workers, int computeThreads, int inputSize, int epochs, long seed)
                throws IOException, InterruptedException {
            List<EpochResult> rows = new ArrayList<>();

            int stepsPerWorkerPerEpoch = Math.max(1, inputSize / (workers * MINI_BATCH_SIZE));
            int minStepsNeededForShutdown = (int) Math.ceil((double) MASTER_TARGET_UPDATES_FALLBACK / workers);
            int steps = Math.max(stepsPerWorkerPerEpoch, minStepsNeededForShutdown);

            for (int epoch = 1; epoch <= epochs; epoch++) {
                Process master = startJavaProcess(List.of(
                        "com.distributed.mlp.Master",
                        String.valueOf(MASTER_PORT),
                        String.valueOf(workers),
                        String.valueOf(steps),
                        String.valueOf(seed + epoch),
                        "false"),
                    javaProps(computeThreads));

                pause(1500L);

                List<Process> workerProcesses = new ArrayList<>();
                for (int workerId = 0; workerId < workers; workerId++) {
                    Process worker = startJavaProcess(List.of(
                            "com.distributed.mlp.Worker",
                            "127.0.0.1",
                            String.valueOf(MASTER_PORT),
                            String.valueOf(workerId),
                            String.valueOf(workers),
                            String.valueOf(steps),
                            String.valueOf(seed + epoch + workerId)),
                            javaProps(computeThreads));
                    workerProcesses.add(worker);
                }

                pause(STARTUP_GRACE_MS);
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

        private static Process startJavaProcess(List<String> classAndArgs, List<String> jvmArgs) throws IOException {
            List<String> cmd = new ArrayList<>();
            cmd.add(getJavaExecutable());
            cmd.addAll(jvmArgs);
            cmd.add("-cp");
            cmd.add(resolveRuntimeClasspath());
            cmd.addAll(classAndArgs);

            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            pb.inheritIO();
            return pb.start();
        }

        private static List<String> javaProps(int computeThreads) {
            List<String> props = new ArrayList<>();
            props.add("-Dmlp.compressGradients=true");
            props.add("-Dmlp.pullEvery=" + DEFAULT_PULL_EVERY);
            props.add("-Dmlp.computeThreads=" + computeThreads);
            props.add("-Dmlp.ioThreads=1");
            int maxSamples = resolveMaxSamples();
            if (maxSamples > 0) {
                props.add("-D" + MAX_SAMPLES_PROP + "=" + maxSamples);
            }
            return props;
        }

        private static int threadsPerWorker(int workers) {
            // int logicalCores = Runtime.getRuntime().availableProcessors();
            // return Math.max(1, logicalCores / Math.max(1, workers));
            return 1;
        }

        private static void pause(long millis) throws InterruptedException {
            Thread.sleep(millis);
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

        private static void configureMaxSamples(String[] args) {
            if (args.length == 0 || args[0] == null || args[0].isBlank()) {
                return;
            }
            int maxSamples = Integer.parseInt(args[0].trim());
            if (maxSamples <= 0) {
                throw new IllegalArgumentException("max samples must be > 0");
            }
            System.setProperty(MAX_SAMPLES_PROP, String.valueOf(maxSamples));
        }

        private static int resolveMaxSamples() {
            String value = System.getProperty(MAX_SAMPLES_PROP);
            if (value == null || value.isBlank()) {
                value = System.getenv("MLP_MAX_SAMPLES");
            }
            if (value == null || value.isBlank()) {
                return 0;
            }
            try {
                return Integer.parseInt(value.trim());
            } catch (NumberFormatException e) {
                return 0;
            }
        }

        private static int effectiveSampleCount(int requested) {
            int maxSamples = resolveMaxSamples();
            if (maxSamples <= 0) {
                return requested;
            }
            return Math.min(requested, maxSamples);
        }

        private record EpochResult(int epoch, double wallSec, double loss, double accuracy) {
        }
    }
