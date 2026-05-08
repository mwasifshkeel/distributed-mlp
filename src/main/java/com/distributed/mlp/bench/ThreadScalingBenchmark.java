package com.distributed.mlp.bench;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
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
import java.util.stream.Collectors;

import javax.imageio.ImageIO;

/**
 * Benchmarks per-worker thread scaling for distributed training and appends
 * results into results/raw.csv. Also writes thread_scaling.csv and plots.
 */
public final class ThreadScalingBenchmark {
    private static final int[] DEFAULT_THREAD_CONFIGS = {1, 2, 4, 8};
    private static final int[] DEFAULT_INPUT_SIZES = {10_000};
    private static final int DEFAULT_WORKERS = 3;
    private static final int DEFAULT_EPOCHS = 2;
    private static final int MASTER_PORT = 9000;
    private static final int MASTER_TARGET_UPDATES_FALLBACK = 100;
    private static final int MINI_BATCH_SIZE = 128;
    private static final long BASE_SEED = 42L;
    private static final int MIN_STEPS_PER_WORKER = 10;
    private static final int STARTUP_GRACE_MS = 200;
    private static final int DEFAULT_PULL_EVERY = 10;
    private static final String MAX_SAMPLES_PROP = "mlp.maxSamples";

    private static final int WIDTH = 1200;
    private static final int HEIGHT = 700;

    private static final Path RESULTS_DIR = Path.of("results");
    private static final Path RAW_CSV = RESULTS_DIR.resolve("raw.csv");
    private static final Path THREAD_CSV = RESULTS_DIR.resolve("thread_scaling.csv");
    private static final Path PLOTS_DIR = RESULTS_DIR.resolve("plots");

    private ThreadScalingBenchmark() {
    }

    public static void main(String[] args) {
        try {
            Config cfg = Config.fromArgs(args);
            Files.createDirectories(RESULTS_DIR);
            Files.createDirectories(PLOTS_DIR);
            ensureRawHeader();

            List<ResultRow> rows = runThreadScaling(cfg);
            appendRawCsv(rows);
            writeThreadCsv(rows, cfg);
            plotThreadScaling(rows, cfg);

            System.out.println("Thread scaling benchmark complete.");
            System.out.println("  Appended to: " + RAW_CSV.toAbsolutePath());
            System.out.println("  Thread CSV : " + THREAD_CSV.toAbsolutePath());
            System.out.println("  Plots      : " + PLOTS_DIR.toAbsolutePath());
        } catch (Exception e) {
            System.err.println("ThreadScalingBenchmark failed: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static List<ResultRow> runThreadScaling(Config cfg) throws IOException, InterruptedException {
        List<ResultRow> rows = new ArrayList<>();
        
        for (int inputSize : cfg.inputSizes) {
            int effectiveInputSize = effectiveSampleCount(inputSize);
            System.out.printf("[ThreadScaling] Running with input_size=%d (effective=%d)%n", 
                inputSize, effectiveInputSize);
            
            for (int threads : cfg.threadConfigs) {
                System.out.printf(
                        Locale.ROOT,
                        "[ThreadScaling] workers=%d threads=%d input=%d epochs=%d%n",
                        cfg.workers,
                        threads,
                        effectiveInputSize,
                        cfg.epochs);

                List<EpochResult> epochs = runAsyncSgd(cfg.workers, effectiveInputSize, cfg.epochs, threads);
                for (EpochResult epoch : epochs) {
                    rows.add(new ResultRow(threads, effectiveInputSize, epoch.epoch(), epoch.wallSec()));
                }
            }
        }
        return rows;
    }

    private static List<EpochResult> runAsyncSgd(int workers, int inputSize, int epochs, int computeThreads)
            throws IOException, InterruptedException {
        List<EpochResult> rows = new ArrayList<>();
        int stepsPerWorkerPerEpoch = Math.max(1, inputSize / (workers * MINI_BATCH_SIZE));
        int minStepsNeededForShutdown = (int) Math.ceil((double) MASTER_TARGET_UPDATES_FALLBACK / workers);
        int steps = Math.max(Math.max(stepsPerWorkerPerEpoch, minStepsNeededForShutdown), MIN_STEPS_PER_WORKER);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            Process master = startJavaProcess(List.of(
                    "com.distributed.mlp.Master",
                    String.valueOf(MASTER_PORT),
                    String.valueOf(workers),
                    String.valueOf(steps),
                    String.valueOf(BASE_SEED + inputSize + epoch),
                    "false"),
                    javaProps(computeThreads, inputSize));

            List<Process> workerProcesses = new ArrayList<>();
            for (int workerId = 0; workerId < workers; workerId++) {
                Process worker = startJavaProcess(List.of(
                        "com.distributed.mlp.Worker",
                        "127.0.0.1",
                        String.valueOf(MASTER_PORT),
                        String.valueOf(workerId),
                        String.valueOf(workers),
                        String.valueOf(steps),
                        String.valueOf(BASE_SEED + inputSize + epoch + workerId)),
                        javaProps(computeThreads, inputSize));
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
                throw new IOException("Master exited with non-zero code " + masterExit
                        + " for threads=" + computeThreads);
            }

            rows.add(new EpochResult(epoch, wallSec));
            System.out.printf(
                    Locale.ROOT,
                    "[ThreadScaling] threads=%d input=%d epoch=%d wall_sec=%.3f%n",
                    computeThreads,
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

    private static List<String> javaProps(int computeThreads, int inputSize) {
        List<String> props = new ArrayList<>();
        props.add("-Dmlp.compressGradients=true");
        props.add("-Dmlp.pullEvery=" + DEFAULT_PULL_EVERY);
        props.add("-Dmlp.computeThreads=" + computeThreads);
        props.add("-Dmlp.ioThreads=1");
        int maxSamples = resolveMaxSamples();
        if (maxSamples > 0) {
            props.add("-D" + MAX_SAMPLES_PROP + "=" + Math.min(inputSize, maxSamples));
        }
        return props;
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

    private static void ensureRawHeader() throws IOException {
        if (Files.exists(RAW_CSV)) {
            return;
        }
        String header = "mode,workers,threads,input_size,epoch,wall_sec" + System.lineSeparator();
        Files.writeString(
                RAW_CSV,
                header,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE);
    }

    private static void appendRawCsv(List<ResultRow> rows) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (ResultRow row : rows) {
            sb.append(String.format(
                    Locale.ROOT,
                    "%s,%d,%d,%d,%d,%.6f%n",
                    "async_sgd_thread_scaling",
                    DEFAULT_WORKERS,
                    row.threads(),
                    row.inputSize(),
                    row.epoch(),
                    row.wallSec()));
        }

        Files.writeString(
                RAW_CSV,
                sb.toString(),
                StandardCharsets.UTF_8,
                StandardOpenOption.APPEND,
                StandardOpenOption.CREATE);
    }

    private static void writeThreadCsv(List<ResultRow> rows, Config cfg) throws IOException {
        for (int inputSize : cfg.inputSizes) {
            int effectiveInputSize = effectiveSampleCount(inputSize);
            List<ResultRow> sizeRows = rows.stream()
                    .filter(r -> r.inputSize() == effectiveInputSize)
                    .collect(Collectors.toList());
            
            if (sizeRows.isEmpty()) continue;
            
            List<Integer> threads = sizeRows.stream()
                    .map(ResultRow::threads)
                    .distinct()
                    .sorted()
                    .collect(Collectors.toList());

            double tBaseline = meanWall(sizeRows, threads.get(0));
            StringBuilder sb = new StringBuilder();
            
            // Append header only for first input size or create new file
            boolean fileExists = Files.exists(THREAD_CSV);
            if (!fileExists) {
                sb.append("workers,threads,input_size,mean_wall_sec,speedup,efficiency")
                        .append(System.lineSeparator());
            }

            for (int t : threads) {
                double tPar = meanWall(sizeRows, t);
                double speedup = tBaseline / tPar;
                double efficiency = speedup / t;
                sb.append(String.format(
                        Locale.ROOT,
                        "%d,%d,%d,%.6f,%.6f,%.6f%n",
                        cfg.workers,
                        t,
                        effectiveInputSize,
                        tPar,
                        speedup,
                        efficiency));
            }

            Files.writeString(
                    THREAD_CSV,
                    sb.toString(),
                    StandardCharsets.UTF_8,
                    fileExists ? StandardOpenOption.APPEND : StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE);
        }
    }

    private static void plotThreadScaling(List<ResultRow> rows, Config cfg) throws IOException {
        for (int inputSize : cfg.inputSizes) {
            int effectiveInputSize = effectiveSampleCount(inputSize);
            List<ResultRow> sizeRows = rows.stream()
                    .filter(r -> r.inputSize() == effectiveInputSize)
                    .collect(Collectors.toList());
            
            if (sizeRows.isEmpty()) continue;
            
            List<Integer> threads = sizeRows.stream()
                    .map(ResultRow::threads)
                    .distinct()
                    .sorted()
                    .collect(Collectors.toList());

            double tBaseline = meanWall(sizeRows, threads.get(0));
            List<Double> x = threads.stream().map(Integer::doubleValue).collect(Collectors.toList());
            List<Double> wall = threads.stream().map(t -> meanWall(sizeRows, t)).collect(Collectors.toList());
            List<Double> speedup = threads.stream().map(t -> tBaseline / meanWall(sizeRows, t)).collect(Collectors.toList());

            String suffix = "_input" + effectiveInputSize;
            
            lineChart(
                    "Thread Scaling Wall Time (workers=" + cfg.workers + ", input=" + effectiveInputSize + ")",
                    "compute_threads",
                    "wall_sec",
                    x,
                    wall,
                    PLOTS_DIR.resolve("thread_scaling_wall_sec" + suffix + ".png"));

            lineChart(
                    "Thread Scaling Speedup (workers=" + cfg.workers + ", input=" + effectiveInputSize + ")",
                    "compute_threads",
                    "speedup",
                    x,
                    speedup,
                    PLOTS_DIR.resolve("thread_scaling_speedup" + suffix + ".png"));
        }
    }

    private static double meanWall(List<ResultRow> rows, int threads) {
        return rows.stream()
                .filter(r -> r.threads() == threads)
                .mapToDouble(ResultRow::wallSec)
                .average()
                .orElse(Double.NaN);
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

    private static void lineChart(String title, String xLabel, String yLabel,
                                  List<Double> x, List<Double> y, Path outFile) throws IOException {
        BufferedImage img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = img.createGraphics();
        setupGraphics(g);

        int left = 90;
        int right = 40;
        int top = 60;
        int bottom = 80;
        int w = WIDTH - left - right;
        int h = HEIGHT - top - bottom;

        g.setColor(Color.WHITE);
        g.fillRect(0, 0, WIDTH, HEIGHT);

        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.BOLD, 18));
        g.drawString(title, left, 30);

        double xMin = x.stream().min(Double::compare).orElse(0.0);
        double xMax = x.stream().max(Double::compare).orElse(1.0);
        if (xMax == xMin) xMax = xMin + 1.0;

        double yMin = y.stream().min(Double::compare).orElse(0.0);
        double yMax = y.stream().max(Double::compare).orElse(1.0);
        if (yMax == yMin) yMax = yMin + 1.0;

        List<Double> xTicks = buildIntegerTicks(xMin, xMax);
        drawLineGrid(g, left, top, w, h, xMin, xMax, yMin, yMax, xTicks);

        g.setFont(new Font("SansSerif", Font.PLAIN, 14));
        g.drawString(xLabel, left + w / 2 - 30, HEIGHT - 30);
        g.drawString(yLabel, 10, top + h / 2);

        g.setColor(new Color(31, 119, 180));
        g.setStroke(new BasicStroke(3.0f));
        for (int i = 0; i < x.size() - 1 && i < y.size() - 1; i++) {
            int x1 = left + (int) ((x.get(i) - xMin) / (xMax - xMin) * w);
            int y1 = top + h - (int) ((y.get(i) - yMin) / (yMax - yMin) * h);
            int x2 = left + (int) ((x.get(i + 1) - xMin) / (xMax - xMin) * w);
            int y2 = top + h - (int) ((y.get(i + 1) - yMin) / (yMax - yMin) * h);
            g.drawLine(x1, y1, x2, y2);
        }

        int r = 4;
        for (int i = 0; i < x.size() && i < y.size(); i++) {
            int cx = left + (int) ((x.get(i) - xMin) / (xMax - xMin) * w);
            int cy = top + h - (int) ((y.get(i) - yMin) / (yMax - yMin) * h);
            g.fillOval(cx - r, cy - r, r * 2, r * 2);
        }

        g.dispose();
        ImageIO.write(img, "png", outFile.toFile());
    }

    private static void setupGraphics(Graphics2D g) {
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setStroke(new BasicStroke(1.0f));
    }

    private static void drawLineGrid(Graphics2D g, int left, int top, int w, int h,
                                     double xMin, double xMax, double yMin, double yMax,
                                     List<Double> xTicks) {
        int yTicks = 6;

        g.setColor(new Color(230, 230, 230));
        for (double v : xTicks) {
            int x = left + (int) ((v - xMin) / (xMax - xMin) * w);
            g.drawLine(x, top, x, top + h);
        }
        for (int i = 0; i <= yTicks; i++) {
            double v = yMin + (yMax - yMin) * i / yTicks;
            int y = top + h - (int) ((v - yMin) / (yMax - yMin) * h);
            g.drawLine(left, y, left + w, y);
        }

        g.setColor(Color.DARK_GRAY);
        g.drawRect(left, top, w, h);

        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        double xSpan = xMax - xMin;
        double ySpan = yMax - yMin;
        for (double v : xTicks) {
            int x = left + (int) ((v - xMin) / xSpan * w);
            g.drawString(formatTick(v, xSpan), x - 10, top + h + 20);
        }
        for (int i = 0; i <= yTicks; i++) {
            double v = yMin + ySpan * i / yTicks;
            int y = top + h - (int) ((v - yMin) / ySpan * h);
            g.drawString(formatTick(v, ySpan), 15, y + 4);
        }
    }

    private static List<Double> buildIntegerTicks(double min, double max) {
        List<Double> ticks = new ArrayList<>();
        int start = (int) Math.ceil(min);
        int end = (int) Math.floor(max);
        for (int v = start; v <= end; v++) {
            ticks.add((double) v);
        }
        if (ticks.isEmpty()) {
            ticks.add(min);
            if (max != min) {
                ticks.add(max);
            }
        }
        return ticks;
    }

    private static String formatTick(double value, double span) {
        if (Math.abs(value - Math.rint(value)) < 1e-6) {
            return String.format(Locale.ROOT, "%.0f", value);
        }
        if (span < 1.0) {
            return String.format(Locale.ROOT, "%.3f", value);
        }
        if (span < 10.0) {
            return String.format(Locale.ROOT, "%.2f", value);
        }
        if (span < 100.0) {
            return String.format(Locale.ROOT, "%.1f", value);
        }
        return String.format(Locale.ROOT, "%.0f", value);
    }

    private static void destroyAll(List<Process> processes) {
        for (Process p : processes) {
            if (p.isAlive()) {
                p.destroyForcibly();
            }
        }
    }

    private record EpochResult(int epoch, double wallSec) {
    }

    private record ResultRow(int threads, int inputSize, int epoch, double wallSec) {
    }

    private record Config(int workers, List<Integer> inputSizes, int epochs, List<Integer> threadConfigs) {
        private static Config fromArgs(String[] args) {
            int workers = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_WORKERS;
            List<Integer> inputSizes = args.length > 1 ? parseIntegers(args[1]) : toList(DEFAULT_INPUT_SIZES);
            int epochs = args.length > 2 ? Integer.parseInt(args[2]) : DEFAULT_EPOCHS;
            List<Integer> threadConfigs = args.length > 3
                    ? parseIntegers(args[3])
                    : toList(DEFAULT_THREAD_CONFIGS);
            return new Config(workers, inputSizes, epochs, threadConfigs);
        }

        private static List<Integer> parseIntegers(String csv) {
            return List.of(csv.split(",")).stream()
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .map(Integer::parseInt)
                    .sorted()
                    .collect(Collectors.toList());
        }

        private static List<Integer> toList(int[] values) {
            List<Integer> out = new ArrayList<>();
            for (int v : values) {
                out.add(v);
            }
            return out;
        }
    }
}