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
    private static final int DEFAULT_WORKERS = 3;
    private static final int DEFAULT_EPOCHS = 2;
    private static final int DEFAULT_INPUT_SIZE = 75_750;
    private static final int MASTER_PORT = 9000;
    private static final int MASTER_TARGET_UPDATES_FALLBACK = 100;
    private static final int MINI_BATCH_SIZE = 32;
    private static final long BASE_SEED = 42L;

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
            appendRawCsv(rows, cfg.inputSize, cfg.workers);
            writeThreadCsv(rows, cfg.inputSize, cfg.workers);
            plotThreadScaling(rows, cfg.inputSize, cfg.workers);

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
        for (int threads : cfg.threadConfigs) {
            System.out.printf(
                    Locale.ROOT,
                    "[ThreadScaling] workers=%d threads=%d input=%d epochs=%d%n",
                    cfg.workers,
                    threads,
                    cfg.inputSize,
                    cfg.epochs);

            List<EpochResult> epochs = runAsyncSgd(cfg.workers, cfg.inputSize, cfg.epochs, threads);
            for (EpochResult epoch : epochs) {
                rows.add(new ResultRow(threads, epoch.epoch(), epoch.wallSec()));
            }
        }
        return rows;
    }

    private static List<EpochResult> runAsyncSgd(int workers, int inputSize, int epochs, int computeThreads)
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
                    String.valueOf(BASE_SEED + inputSize + epoch),
                    "false"),
                    List.of());

            Thread.sleep(1500L);

            List<Process> workerProcesses = new ArrayList<>();
            for (int workerId = 0; workerId < workers; workerId++) {
                List<String> jvmArgs = List.of(
                        "-Dmlp.computeThreads=" + computeThreads,
                        "-Dmlp.ioThreads=1");
                Process worker = startJavaProcess(List.of(
                        "com.distributed.mlp.Worker",
                        "127.0.0.1",
                        String.valueOf(MASTER_PORT),
                        String.valueOf(workerId),
                        String.valueOf(workers),
                        String.valueOf(steps),
                        String.valueOf(BASE_SEED + inputSize + epoch + workerId)),
                        jvmArgs);
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
                    "[ThreadScaling] threads=%d epoch=%d wall_sec=%.3f%n",
                    computeThreads,
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
        String header = "mode,workers,threads,input_size,epoch,wall_sec,loss,accuracy" + System.lineSeparator();
        Files.writeString(
                RAW_CSV,
                header,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE);
    }

    private static void appendRawCsv(List<ResultRow> rows, int inputSize, int workers) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (ResultRow row : rows) {
            sb.append(String.format(
                    Locale.ROOT,
                    "%s,%d,%d,%d,%d,%.6f,NaN,NaN%n",
                    "async_sgd_thread_scaling",
                    workers,
                    row.threads(),
                    inputSize,
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

    private static void writeThreadCsv(List<ResultRow> rows, int inputSize, int workers) throws IOException {
        List<Integer> threads = rows.stream()
                .map(ResultRow::threads)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        double tBaseline = meanWall(rows, threads.get(0));
        StringBuilder sb = new StringBuilder();
        sb.append("workers,threads,input_size,mean_wall_sec,speedup,efficiency")
                .append(System.lineSeparator());

        for (int t : threads) {
            double tPar = meanWall(rows, t);
            double speedup = tBaseline / tPar;
            double efficiency = speedup / t;
            sb.append(String.format(
                    Locale.ROOT,
                    "%d,%d,%d,%.6f,%.6f,%.6f%n",
                    workers,
                    t,
                    inputSize,
                    tPar,
                    speedup,
                    efficiency));
        }

        Files.writeString(
                THREAD_CSV,
                sb.toString(),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static void plotThreadScaling(List<ResultRow> rows, int inputSize, int workers) throws IOException {
        List<Integer> threads = rows.stream()
                .map(ResultRow::threads)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        double tBaseline = meanWall(rows, threads.get(0));
        List<Double> x = threads.stream().map(Integer::doubleValue).collect(Collectors.toList());
        List<Double> wall = threads.stream().map(t -> meanWall(rows, t)).collect(Collectors.toList());
        List<Double> speedup = threads.stream().map(t -> tBaseline / meanWall(rows, t)).collect(Collectors.toList());

        lineChart(
                "Thread Scaling Wall Time (workers=" + workers + ")",
                "compute_threads",
                "wall_sec",
                x,
                wall,
                PLOTS_DIR.resolve("thread_scaling_wall_sec.png"));

        lineChart(
                "Thread Scaling Speedup (workers=" + workers + ")",
                "compute_threads",
                "speedup",
                x,
                speedup,
                PLOTS_DIR.resolve("thread_scaling_speedup.png"));
    }

    private static double meanWall(List<ResultRow> rows, int threads) {
        return rows.stream()
                .filter(r -> r.threads() == threads)
                .mapToDouble(ResultRow::wallSec)
                .average()
                .orElse(Double.NaN);
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

    private record ResultRow(int threads, int epoch, double wallSec) {
    }

    private record Config(int workers, int inputSize, int epochs, List<Integer> threadConfigs) {
        private static Config fromArgs(String[] args) {
            int workers = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_WORKERS;
            int inputSize = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_INPUT_SIZE;
            int epochs = args.length > 2 ? Integer.parseInt(args[2]) : DEFAULT_EPOCHS;
            List<Integer> threadConfigs = args.length > 3
                    ? parseThreads(args[3])
                    : toList(DEFAULT_THREAD_CONFIGS);
            return new Config(workers, inputSize, epochs, threadConfigs);
        }

        private static List<Integer> parseThreads(String csv) {
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
