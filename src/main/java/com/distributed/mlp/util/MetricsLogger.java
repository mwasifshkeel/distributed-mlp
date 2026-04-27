package com.distributed.mlp.util;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Thread-safe CSV logger for per-epoch metrics.
 */
public final class MetricsLogger {
    private static final String[] DEFAULT_HEADER = {
        "timestamp_ms",
        "component",
        "worker_id",
        "epoch",
        "loss",
        "accuracy",
        "wall_sec"
    };

    private final Path csvPath;
    private final ReentrantLock writerLock = new ReentrantLock();

    /**
     * Creates a logger that writes to results/metrics.csv.
     */
    public MetricsLogger() throws IOException {
        this(Path.of("results", "metrics.csv"));
    }

    /**
     * Creates a logger that writes to a custom CSV path.
     */
    public MetricsLogger(Path csvPath) throws IOException {
        if (csvPath == null) {
            throw new IllegalArgumentException("csvPath must not be null");
        }
        this.csvPath = csvPath;
        initIfNeeded();
    }

    /**
     * Appends a CSV row atomically.
     */
    public void append(String... cols) throws IOException {
        if (cols == null || cols.length == 0) {
            throw new IllegalArgumentException("cols must not be empty");
        }

        writerLock.lock();
        try {
            String line = toCsvLine(cols) + System.lineSeparator();
            Files.writeString(
                    csvPath,
                    line,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND,
                    StandardOpenOption.WRITE);
        } finally {
            writerLock.unlock();
        }
    }

    /**
     * Convenience helper for standard epoch metrics rows.
     */
    public void appendEpochMetrics(
            String component,
            int workerId,
            int epoch,
            double loss,
            double accuracy,
            double wallSec) throws IOException {
        append(
                String.valueOf(System.currentTimeMillis()),
                sanitize(component),
                String.valueOf(workerId),
                String.valueOf(epoch),
                String.format(Locale.ROOT, "%.6f", loss),
                String.format(Locale.ROOT, "%.6f", accuracy),
                String.format(Locale.ROOT, "%.6f", wallSec));
    }

    public Path getCsvPath() {
        return csvPath;
    }

    private void initIfNeeded() throws IOException {
        Path parent = csvPath.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        writerLock.lock();
        try {
            if (!Files.exists(csvPath) || Files.size(csvPath) == 0L) {
                String header = toCsvLine(DEFAULT_HEADER) + System.lineSeparator();
                Files.writeString(
                        csvPath,
                        header,
                        StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.WRITE);
            }
        } finally {
            writerLock.unlock();
        }
    }

    private static String toCsvLine(String... cols) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cols.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(escapeCsv(cols[i] == null ? "" : cols[i]));
        }
        return sb.toString();
    }

    private static String sanitize(String value) {
        return value == null ? "" : value;
    }

    private static String escapeCsv(String value) {
        boolean needsQuotes = value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r");
        if (!needsQuotes) {
            return value;
        }
        return '"' + value.replace("\"", "\"\"") + '"';
    }
}
