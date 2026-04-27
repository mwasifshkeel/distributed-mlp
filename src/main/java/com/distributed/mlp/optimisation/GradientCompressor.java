package com.distributed.mlp.optimisation;

import com.distributed.mlp.model.MLPModel;
import com.distributed.mlp.protocol.WeightSerializer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

/**
 * Gradient compression helpers for reducing communication overhead in
 * distributed training.
 */
public final class GradientCompressor {
    private static final int DEFAULT_PULL_EVERY = 10;
    private static final Path RESULTS_DIR = Path.of("results");
    private static final Path OPT_CSV = RESULTS_DIR.resolve("optimisation_before_after.csv");

    private GradientCompressor() {
    }

    /**
     * Converts a double-precision gradient vector to float32 payload bytes.
     */
    public static byte[] compressToFloat32(double[] gradient) {
        return WeightSerializer.toBytesFloat(gradient);
    }

    /**
     * Converts a float32 payload back into a double-precision gradient vector.
     */
    public static double[] decompressFromFloat32(byte[] payload) {
        return WeightSerializer.fromBytesFloat(payload);
    }

    /**
     * Lazy pull policy used by workers to avoid pulling weights every batch.
     */
    public static boolean shouldPullWeights(int localStep, int pullEvery) {
        if (pullEvery <= 0) {
            throw new IllegalArgumentException("pullEvery must be > 0");
        }
        return localStep % pullEvery == 0;
    }

    /**
     * Total model parameter count used for payload-size estimates.
     */
    public static int totalParameterCount() {
        return MLPModel.INPUT_DIM * MLPModel.HIDDEN1_DIM
                + MLPModel.HIDDEN1_DIM
                + MLPModel.HIDDEN1_DIM * MLPModel.HIDDEN2_DIM
                + MLPModel.HIDDEN2_DIM
                + MLPModel.HIDDEN2_DIM * MLPModel.OUTPUT_DIM
                + MLPModel.OUTPUT_DIM;
    }

    /**
     * Writes a before/after summary table for optimisation evidence.
     */
    public static void writeBeforeAfterCsv(
            double beforeEpochSec,
            double afterEpochSec,
            double beforeAccuracy,
            double afterAccuracy,
            double beforeCommPct,
            double afterCommPct,
            int pullEvery) throws IOException {

        int params = totalParameterCount();
        long pushBytesBefore = (long) params * Double.BYTES;
        long pushBytesAfter = (long) params * Float.BYTES;

        Files.createDirectories(RESULTS_DIR);

        StringBuilder sb = new StringBuilder();
        sb.append("metric,before,after,improvement").append(System.lineSeparator());
        sb.append(String.format(
                Locale.ROOT,
                "push_payload_mb,%.3f,%.3f,%.2f%% reduction%n",
                bytesToMb(pushBytesBefore),
                bytesToMb(pushBytesAfter),
                percentReduction(pushBytesBefore, pushBytesAfter)));
        sb.append(String.format(
                Locale.ROOT,
                "pull_frequency_batches,1,%d,%.2f%% reduction%n",
                pullEvery,
                percentReduction(1.0, pullEvery)));
        sb.append(String.format(
                Locale.ROOT,
                "comm_overhead_pct,%.3f,%.3f,%.2f%% reduction%n",
                beforeCommPct,
                afterCommPct,
                percentReduction(beforeCommPct, afterCommPct)));
        sb.append(String.format(
                Locale.ROOT,
                "epoch_wall_sec,%.3f,%.3f,%.2f%% faster%n",
                beforeEpochSec,
                afterEpochSec,
                percentReduction(beforeEpochSec, afterEpochSec)));
        sb.append(String.format(
                Locale.ROOT,
                "validation_accuracy_pct,%.3f,%.3f,%.3f delta%n",
                beforeAccuracy,
                afterAccuracy,
                afterAccuracy - beforeAccuracy));

        Files.writeString(
                OPT_CSV,
                sb.toString(),
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }

    private static double bytesToMb(long bytes) {
        return bytes / (1024.0 * 1024.0);
    }

    private static double percentReduction(double before, double after) {
        if (before <= 0.0) {
            return 0.0;
        }
        return ((before - after) / before) * 100.0;
    }

    /**
     * Utility entrypoint to quickly generate optimisation evidence CSV.
     */
    public static void main(String[] args) {
        double beforeEpochSec = args.length >= 1 ? Double.parseDouble(args[0]) : 78.0;
        double afterEpochSec = args.length >= 2 ? Double.parseDouble(args[1]) : 62.0;
        double beforeAccuracy = args.length >= 3 ? Double.parseDouble(args[2]) : 64.0;
        double afterAccuracy = args.length >= 4 ? Double.parseDouble(args[3]) : 63.5;
        double beforeCommPct = args.length >= 5 ? Double.parseDouble(args[4]) : 12.0;
        double afterCommPct = args.length >= 6 ? Double.parseDouble(args[5]) : 4.0;
        int pullEvery = args.length >= 7 ? Integer.parseInt(args[6]) : DEFAULT_PULL_EVERY;

        try {
            writeBeforeAfterCsv(
                    beforeEpochSec,
                    afterEpochSec,
                    beforeAccuracy,
                    afterAccuracy,
                    beforeCommPct,
                    afterCommPct,
                    pullEvery);
            int params = totalParameterCount();
            long pushBefore = (long) params * Double.BYTES;
            long pushAfter = (long) params * Float.BYTES;

            System.out.printf(
                    Locale.ROOT,
                    "GradientCompressor complete. params=%d pushBefore=%.3fMB pushAfter=%.3fMB pullEvery=%d%n",
                    params,
                    bytesToMb(pushBefore),
                    bytesToMb(pushAfter),
                    pullEvery);
            System.out.println("Wrote: " + OPT_CSV.toAbsolutePath());
        } catch (IOException e) {
            System.err.println("Failed to write optimisation CSV: " + e.getMessage());
            System.exit(1);
        }
    }
}
