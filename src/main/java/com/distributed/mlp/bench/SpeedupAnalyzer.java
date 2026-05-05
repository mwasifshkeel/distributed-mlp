package com.distributed.mlp.bench;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 * Computes speedup/efficiency metrics from benchmark CSV output and writes
 * report-ready artifacts used in the project write-up.
 */
public final class SpeedupAnalyzer {
    private static final Path RESULTS_DIR = Path.of("results");
    private static final Path RAW_CSV = RESULTS_DIR.resolve("raw.csv");
    private static final Path SPEEDUP_TABLE = RESULTS_DIR.resolve("speedup_table.txt");
    private static final Path AMDAHL_CSV = RESULTS_DIR.resolve("amdahl_comparison.csv");

    private static final DecimalFormat DF =
            new DecimalFormat("0.000", DecimalFormatSymbols.getInstance(Locale.ROOT));

    private SpeedupAnalyzer() {
    }

    public static void main(String[] args) {
        try {
            analyze();
        } catch (Exception e) {
            System.err.println("SpeedupAnalyzer failed: " + e.getMessage());
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    public static void analyze() throws IOException {
        if (!Files.exists(RAW_CSV)) {
            throw new IOException("Input CSV not found: " + RAW_CSV.toAbsolutePath());
        }

        List<Row> rows = readRows(RAW_CSV);
        if (rows.isEmpty()) {
            throw new IOException("No benchmark rows found in " + RAW_CSV.toAbsolutePath());
        }

        int targetInputSize = rows.stream().mapToInt(Row::inputSize).max().orElseThrow();
        List<Row> filtered = rows.stream()
                .filter(r -> r.inputSize() == targetInputSize)
                .collect(Collectors.toList());

        if (filtered.isEmpty()) {
            throw new IOException("No rows found for selected input size " + targetInputSize);
        }

        double tSeq = meanWallSec(filtered, "sequential", 1);
        if (Double.isNaN(tSeq) || tSeq <= 0.0) {
            throw new IOException("Sequential baseline is missing or invalid for input size " + targetInputSize);
        }

        List<Integer> workers = filtered.stream()
                .filter(r -> "async_sgd".equalsIgnoreCase(r.mode()))
                .map(Row::workers)
                .distinct()
                .sorted()
                .collect(Collectors.toCollection(ArrayList::new));

        if (!workers.contains(1)) {
            workers.add(0, 1);
        }

        List<ResultPoint> points = new ArrayList<>();
        for (int p : workers) {
            double tPar = p == 1 ? tSeq : meanWallSec(filtered, "async_sgd", p);
            if (Double.isNaN(tPar) || tPar <= 0.0) {
                continue;
            }

            double speedup = tSeq / tPar;
            double efficiency = speedup / p;
            double f = p == 1 ? Double.NaN : estimateParallelFraction(speedup, p);
            double amdahlBound = p == 1 ? 1.0 : amdahlSpeedup(f, p);

            points.add(new ResultPoint(p, tPar, speedup, efficiency, f, amdahlBound));
        }

        if (points.isEmpty()) {
            throw new IOException("No valid speedup points could be computed from raw.csv");
        }

        writeSpeedupTable(points, targetInputSize, tSeq);
        writeAmdahlCsv(points);
        printSummary(points, targetInputSize, tSeq);
    }

    private static List<Row> readRows(Path csvPath) throws IOException {
        List<String> lines = Files.readAllLines(csvPath, StandardCharsets.UTF_8);
        List<Row> rows = new ArrayList<>();

        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }

            String[] cols = line.split(",");
            if (cols.length < 8) {
                continue;
            }

            String mode = cols[0].trim();
            int workers = parseInt(cols[1]);
            int inputSize = parseInt(cols[3]);
            int epoch = parseInt(cols[4]);
            double wallSec = parseDouble(cols[5]);

            if (workers <= 0 || inputSize <= 0 || epoch <= 0 || Double.isNaN(wallSec) || wallSec <= 0.0) {
                continue;
            }

            rows.add(new Row(mode, workers, inputSize, epoch, wallSec));
        }

        return rows;
    }

    private static double meanWallSec(List<Row> rows, String mode, int workers) {
        return rows.stream()
                .filter(r -> mode.equalsIgnoreCase(r.mode()))
                .filter(r -> r.workers() == workers)
                .mapToDouble(Row::wallSec)
                .average()
                .orElse(Double.NaN);
    }

    private static double estimateParallelFraction(double speedup, int workers) {
        double numerator = (1.0 / speedup) - 1.0;
        double denominator = (1.0 / workers) - 1.0;
        if (denominator == 0.0) {
            return Double.NaN;
        }
        return numerator / denominator;
    }

    private static double amdahlSpeedup(double f, int workers) {
        return 1.0 / ((1.0 - f) + (f / workers));
    }

    private static void writeSpeedupTable(List<ResultPoint> points, int inputSize, double tSeq) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("Input size: ").append(inputSize).append(System.lineSeparator());
        sb.append("Sequential baseline T_seq: ").append(DF.format(tSeq)).append(" sec").append(System.lineSeparator());
        sb.append(System.lineSeparator());
        sb.append(String.format(
                Locale.ROOT,
                "%-8s %-12s %-12s %-12s %-12s %-12s%n",
                "workers",
                "T_par(sec)",
                "S(p)",
                "E(p)",
                "f(est)",
                "Amdahl"));

        for (ResultPoint p : points.stream().sorted(Comparator.comparingInt(ResultPoint::workers)).toList()) {
            sb.append(String.format(
                    Locale.ROOT,
                    "%-8d %-12s %-12s %-12s %-12s %-12s%n",
                    p.workers(),
                    DF.format(p.tPar()),
                    DF.format(p.speedup()),
                    DF.format(p.efficiency()),
                    formatMaybeNa(p.f()),
                    DF.format(p.amdahlBound())));
        }

        Files.createDirectories(RESULTS_DIR);
        Files.writeString(
                SPEEDUP_TABLE,
                sb.toString(),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static void writeAmdahlCsv(List<ResultPoint> points) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("workers,measured_speedup,amdahl_estimated,amdahl_f_0_5,amdahl_f_0_9,amdahl_f_0_99")
                .append(System.lineSeparator());

        for (ResultPoint p : points.stream().sorted(Comparator.comparingInt(ResultPoint::workers)).toList()) {
            int workers = p.workers();
            sb.append(String.format(
                    Locale.ROOT,
                    "%d,%.6f,%.6f,%.6f,%.6f,%.6f%n",
                    workers,
                    p.speedup(),
                    p.amdahlBound(),
                    amdahlSpeedup(0.5, workers),
                    amdahlSpeedup(0.9, workers),
                    amdahlSpeedup(0.99, workers)));
        }

        Files.createDirectories(RESULTS_DIR);
        Files.writeString(
                AMDAHL_CSV,
                sb.toString(),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static void printSummary(List<ResultPoint> points, int inputSize, double tSeq) {
        System.out.printf(
                Locale.ROOT,
                "Speedup analysis complete for input_size=%d (T_seq=%.3f sec).%n",
                inputSize,
                tSeq);
        for (ResultPoint p : points.stream().sorted(Comparator.comparingInt(ResultPoint::workers)).toList()) {
            System.out.printf(
                    Locale.ROOT,
                    "p=%d T_par=%.3f S=%.3f E=%.3f f=%s Amdahl=%.3f%n",
                    p.workers(),
                    p.tPar(),
                    p.speedup(),
                    p.efficiency(),
                    formatMaybeNa(p.f()),
                    p.amdahlBound());
        }
        System.out.println("Wrote: " + SPEEDUP_TABLE.toAbsolutePath());
        System.out.println("Wrote: " + AMDAHL_CSV.toAbsolutePath());
    }

    private static String formatMaybeNa(double value) {
        return Double.isNaN(value) ? "N/A" : DF.format(value);
    }

    private static int parseInt(String value) {
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    private static double parseDouble(String value) {
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            return Double.NaN;
        }
    }

    private record Row(String mode, int workers, int inputSize, int epoch, double wallSec) {
    }

    private record ResultPoint(int workers, double tPar, double speedup, double efficiency, double f, double amdahlBound) {
    }
}
