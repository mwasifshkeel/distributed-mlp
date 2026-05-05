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
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;

/**
 * Generates PNG plots from benchmark CSV outputs.
 */
public final class BenchPlotter {
    private static final Path RESULTS_DIR = Path.of("results");
    private static final Path PLOTS_DIR = RESULTS_DIR.resolve("plots");

    private static final Path SEQ_CSV = RESULTS_DIR.resolve("sequential_results.csv");
    private static final Path STRONG_CSV = RESULTS_DIR.resolve("strong_scaling.csv");
    private static final Path WEAK_CSV = RESULTS_DIR.resolve("weak_scaling.csv");
    private static final Path AMDAHL_CSV = RESULTS_DIR.resolve("amdahl_comparison.csv");
    private static final Path SERIAL_CSV = RESULTS_DIR.resolve("serial_bench.csv");
    private static final Path OPT_CSV = RESULTS_DIR.resolve("optimisation_runs.csv");

    private static final int WIDTH = 1200;
    private static final int HEIGHT = 700;

    private BenchPlotter() {
    }

    public static void main(String[] args) {
        try {
            Files.createDirectories(PLOTS_DIR);
            plotSequential();
            plotScaling();
            plotAmdahl();
            plotSerialBench();
            plotOptimization();
            System.out.println("Plots saved under: " + PLOTS_DIR.toAbsolutePath());
        } catch (Exception e) {
            System.err.println("BenchPlotter failed: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static void plotSequential() throws IOException {
        if (!Files.exists(SEQ_CSV)) {
            System.err.println("[BenchPlotter] Missing sequential_results.csv, skipping.");
            return;
        }
        List<String[]> rows = readCsv(SEQ_CSV);
        List<Double> epochs = columnAsDouble(rows, 0);
        List<Double> loss = columnAsDouble(rows, 1);
        List<Double> acc = columnAsDouble(rows, 2);
        List<Double> wall = columnAsDouble(rows, 3);

        lineChart(
                "Sequential Loss by Epoch",
                "epoch",
                "loss",
                epochs,
                List.of(loss),
                List.of("loss"),
                PLOTS_DIR.resolve("sequential_loss.png"));

        lineChart(
                "Sequential Accuracy by Epoch",
                "epoch",
                "accuracy",
                epochs,
                List.of(acc),
                List.of("accuracy"),
                PLOTS_DIR.resolve("sequential_accuracy.png"));

        lineChart(
                "Sequential Wall Time by Epoch",
                "epoch",
                "wall_sec",
                epochs,
                List.of(wall),
                List.of("wall_sec"),
                PLOTS_DIR.resolve("sequential_wall_sec.png"));
    }

    private static void plotScaling() throws IOException {
        if (Files.exists(STRONG_CSV)) {
            List<ScalingRow> strong = readScaling(STRONG_CSV);
            plotScalingSeries(strong, "strong", "strong_speedup.png", "strong_efficiency.png");
        } else {
            System.err.println("[BenchPlotter] Missing strong_scaling.csv, skipping.");
        }

        if (Files.exists(WEAK_CSV)) {
            List<ScalingRow> weak = readScaling(WEAK_CSV);
            plotScalingSeries(weak, "weak", "weak_speedup.png", "weak_efficiency.png");
        } else {
            System.err.println("[BenchPlotter] Missing weak_scaling.csv, skipping.");
        }
    }

    private static void plotScalingSeries(List<ScalingRow> rows, String label,
                                          String speedupOut, String effOut) throws IOException {
        Map<Integer, List<ScalingRow>> byWorkers = rows.stream()
                .collect(Collectors.groupingBy(ScalingRow::workers));

        List<Double> workers = byWorkers.keySet().stream()
                .sorted()
                .map(Integer::doubleValue)
                .collect(Collectors.toList());

        List<Double> avgSpeedup = new ArrayList<>();
        List<Double> avgEfficiency = new ArrayList<>();

        for (double w : workers) {
            List<ScalingRow> group = byWorkers.get((int) w);
            avgSpeedup.add(group.stream().mapToDouble(ScalingRow::speedup).average().orElse(0.0));
            avgEfficiency.add(group.stream().mapToDouble(ScalingRow::efficiency).average().orElse(0.0));
        }

        lineChart(
                "" + label + " scaling speedup",
                "workers",
                "speedup",
                workers,
                List.of(avgSpeedup),
                List.of("speedup"),
                PLOTS_DIR.resolve(speedupOut));

        lineChart(
                "" + label + " scaling efficiency",
                "workers",
                "efficiency",
                workers,
                List.of(avgEfficiency),
                List.of("efficiency"),
                PLOTS_DIR.resolve(effOut));
    }

    private static void plotAmdahl() throws IOException {
        if (!Files.exists(AMDAHL_CSV)) {
            System.err.println("[BenchPlotter] Missing amdahl_comparison.csv, skipping.");
            return;
        }
        List<String[]> rows = readCsv(AMDAHL_CSV);
        List<Double> workers = columnAsDouble(rows, 0);
        List<Double> measured = columnAsDouble(rows, 1);
        List<Double> amdahl = columnAsDouble(rows, 2);
        List<Double> f05 = columnAsDouble(rows, 3);
        List<Double> f09 = columnAsDouble(rows, 4);
        List<Double> f099 = columnAsDouble(rows, 5);

        lineChart(
                "Amdahl Speedup Comparison",
                "workers",
                "speedup",
                workers,
                List.of(measured, amdahl, f05, f09, f099),
                List.of("measured", "estimated", "f=0.5", "f=0.9", "f=0.99"),
                PLOTS_DIR.resolve("amdahl_speedup.png"));
    }

    private static void plotSerialBench() throws IOException {
        if (!Files.exists(SERIAL_CSV)) {
            System.err.println("[BenchPlotter] Missing serial_bench.csv, skipping.");
            return;
        }
        List<String[]> rows = readCsv(SERIAL_CSV);
        List<String> labels = rows.stream().map(r -> r[0].trim()).collect(Collectors.toList());
        List<Double> values = rows.stream().map(r -> parseDouble(r[1])).collect(Collectors.toList());

        barChart(
                "Serializer Benchmark (avg_ms)",
                "method",
                "avg_ms",
                labels,
                values,
                PLOTS_DIR.resolve("serial_bench.png"));
    }

    private static void plotOptimization() throws IOException {
        if (!Files.exists(OPT_CSV)) {
            System.err.println("[BenchPlotter] Missing optimisation_runs.csv, skipping.");
            return;
        }
        List<String[]> rows = readCsv(OPT_CSV);
        List<String> labels = rows.stream().map(r -> r[0].trim()).collect(Collectors.toList());
        List<Double> values = rows.stream().map(r -> parseDouble(r[5])).collect(Collectors.toList());

        barChart(
                "Baseline vs Optimized Wall Time",
                "mode",
                "wall_sec",
                labels,
                values,
                PLOTS_DIR.resolve("optimisation_wall_sec.png"));
    }

    private static List<String[]> readCsv(Path csv) throws IOException {
        List<String> lines = Files.readAllLines(csv, StandardCharsets.UTF_8);
        List<String[]> rows = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) continue;
            rows.add(line.split(","));
        }
        return rows;
    }

    private static List<Double> columnAsDouble(List<String[]> rows, int idx) {
        List<Double> out = new ArrayList<>();
        for (String[] row : rows) {
            if (row.length <= idx) continue;
            out.add(parseDouble(row[idx]));
        }
        return out;
    }

    private static List<ScalingRow> readScaling(Path csv) throws IOException {
        List<String> lines = Files.readAllLines(csv, StandardCharsets.UTF_8);
        List<ScalingRow> out = new ArrayList<>();
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) continue;
            String[] cols = line.split(",");
            if (cols.length < 8) continue;
            out.add(new ScalingRow(
                    cols[0].trim(),
                    parseInt(cols[1]),
                    parseInt(cols[2]),
                    parseInt(cols[3]),
                    parseDouble(cols[4]),
                    parseDouble(cols[5]),
                    parseDouble(cols[6])));
        }
        return out;
    }

    private static void lineChart(String title, String xLabel, String yLabel,
                                  List<Double> x,
                                  List<List<Double>> series,
                                  List<String> labels,
                                  Path outFile) throws IOException {
        BufferedImage img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = img.createGraphics();
        setupGraphics(g);

        int left = 80;
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

        double yMin = series.stream().flatMap(List::stream).min(Double::compare).orElse(0.0);
        double yMax = series.stream().flatMap(List::stream).max(Double::compare).orElse(1.0);
        if (yMax == yMin) yMax = yMin + 1.0;

        g.setColor(Color.DARK_GRAY);
        g.drawRect(left, top, w, h);

        g.setFont(new Font("SansSerif", Font.PLAIN, 14));
        g.drawString(xLabel, left + w / 2 - 20, HEIGHT - 30);
        g.drawString(yLabel, 10, top + h / 2);

        Color[] palette = new Color[]{
                new Color(31, 119, 180),
                new Color(255, 127, 14),
                new Color(44, 160, 44),
                new Color(214, 39, 40),
                new Color(148, 103, 189)
        };

        for (int s = 0; s < series.size(); s++) {
            List<Double> ys = series.get(s);
            g.setColor(palette[s % palette.length]);
            g.setStroke(new BasicStroke(2.5f));
            for (int i = 0; i < x.size() - 1 && i < ys.size() - 1; i++) {
                int x1 = left + (int) ((x.get(i) - xMin) / (xMax - xMin) * w);
                int y1 = top + h - (int) ((ys.get(i) - yMin) / (yMax - yMin) * h);
                int x2 = left + (int) ((x.get(i + 1) - xMin) / (xMax - xMin) * w);
                int y2 = top + h - (int) ((ys.get(i + 1) - yMin) / (yMax - yMin) * h);
                g.drawLine(x1, y1, x2, y2);
            }
        }

        if (!labels.isEmpty()) {
            drawLegend(g, labels, palette, left + 10, top + 10);
        }

        g.dispose();
        ImageIO.write(img, "png", outFile.toFile());
    }

    private static void barChart(String title, String xLabel, String yLabel,
                                 List<String> labels, List<Double> values, Path outFile)
            throws IOException {
        BufferedImage img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = img.createGraphics();
        setupGraphics(g);

        int left = 100;
        int right = 40;
        int top = 60;
        int bottom = 100;
        int w = WIDTH - left - right;
        int h = HEIGHT - top - bottom;

        g.setColor(Color.WHITE);
        g.fillRect(0, 0, WIDTH, HEIGHT);

        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.BOLD, 18));
        g.drawString(title, left, 30);

        double max = values.stream().max(Double::compare).orElse(1.0);
        if (max == 0.0) max = 1.0;

        g.setColor(Color.DARK_GRAY);
        g.drawRect(left, top, w, h);

        int barCount = Math.max(1, values.size());
        int barWidth = w / barCount - 10;
        for (int i = 0; i < values.size(); i++) {
            double v = values.get(i);
            int barHeight = (int) ((v / max) * (h - 10));
            int x = left + i * (w / barCount) + 5;
            int y = top + h - barHeight;

            g.setColor(new Color(31, 119, 180));
            g.fillRect(x, y, barWidth, barHeight);
            g.setColor(Color.BLACK);
            g.drawRect(x, y, barWidth, barHeight);

            g.setFont(new Font("SansSerif", Font.PLAIN, 12));
            g.drawString(labels.get(i), x, top + h + 20);
            g.drawString(String.format(Locale.ROOT, "%.2f", v), x, y - 6);
        }

        g.setFont(new Font("SansSerif", Font.PLAIN, 14));
        g.drawString(xLabel, left + w / 2 - 20, HEIGHT - 30);
        g.drawString(yLabel, 10, top + h / 2);

        g.dispose();
        ImageIO.write(img, "png", outFile.toFile());
    }

    private static void setupGraphics(Graphics2D g) {
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setStroke(new BasicStroke(1.0f));
    }

    private static void drawLegend(Graphics2D g, List<String> labels, Color[] colors, int x, int y) {
        g.setFont(new Font("SansSerif", Font.PLAIN, 12));
        int box = 12;
        int pad = 6;
        for (int i = 0; i < labels.size(); i++) {
            g.setColor(colors[i % colors.length]);
            g.fillRect(x, y + i * 18, box, box);
            g.setColor(Color.BLACK);
            g.drawRect(x, y + i * 18, box, box);
            g.drawString(labels.get(i), x + box + pad, y + i * 18 + box - 2);
        }
    }

    private static int parseInt(String value) {
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private static double parseDouble(String value) {
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            return Double.NaN;
        }
    }

    private record ScalingRow(
            String experiment,
            int workers,
            int inputSize,
            int epoch,
            double wallSec,
            double speedup,
            double efficiency) {
    }
}
