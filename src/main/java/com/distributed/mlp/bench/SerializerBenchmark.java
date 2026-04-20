package com.distributed.mlp.bench;

import com.distributed.mlp.protocol.WeightSerializer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.Random;

/**
 * Benchmarks serialization of a 500k double array using ObjectOutputStream vs ByteBuffer serializer.
 */
public final class SerializerBenchmark {
    private static final int ARRAY_SIZE = 500_000;
    private static final int WARMUP_ROUNDS = 2;
    private static final int MEASURE_ROUNDS = 5;

    private SerializerBenchmark() {
    }

    public static void main(String[] args) throws IOException {
        double[] weights = createRandomArray(ARRAY_SIZE, 42L);

        for (int i = 0; i < WARMUP_ROUNDS; i++) {
            serializeWithObjectOutputStream(weights);
            WeightSerializer.toBytesDouble(weights);
        }

        long objectTimeNanos = 0L;
        byte[] objectBytes = null;
        for (int i = 0; i < MEASURE_ROUNDS; i++) {
            long start = System.nanoTime();
            objectBytes = serializeWithObjectOutputStream(weights);
            objectTimeNanos += System.nanoTime() - start;
        }

        long byteBufferTimeNanos = 0L;
        byte[] byteBufferBytes = null;
        for (int i = 0; i < MEASURE_ROUNDS; i++) {
            long start = System.nanoTime();
            byteBufferBytes = WeightSerializer.toBytesDouble(weights);
            byteBufferTimeNanos += System.nanoTime() - start;
        }

        double objectAvgMs = nanosToMillis(objectTimeNanos / MEASURE_ROUNDS);
        double byteBufferAvgMs = nanosToMillis(byteBufferTimeNanos / MEASURE_ROUNDS);
        double speedup = objectAvgMs / byteBufferAvgMs;

        System.out.printf(Locale.ROOT,
                "Serializer benchmark (array=%d, rounds=%d)%n", ARRAY_SIZE, MEASURE_ROUNDS);
        System.out.printf(Locale.ROOT,
                "ObjectOutputStream avg: %.3f ms, payload=%d bytes%n",
                objectAvgMs, objectBytes == null ? -1 : objectBytes.length);
        System.out.printf(Locale.ROOT,
                "ByteBuffer avg: %.3f ms, payload=%d bytes%n",
                byteBufferAvgMs, byteBufferBytes == null ? -1 : byteBufferBytes.length);
        System.out.printf(Locale.ROOT, "Speedup (Object/ByteBuffer): %.2fx%n", speedup);

        writeCsv(objectAvgMs, byteBufferAvgMs,
                objectBytes == null ? -1 : objectBytes.length,
                byteBufferBytes == null ? -1 : byteBufferBytes.length,
                speedup);
    }

    private static byte[] serializeWithObjectOutputStream(double[] values) throws IOException {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream(values.length * Double.BYTES);
             ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(values);
            oos.flush();
            return baos.toByteArray();
        }
    }

    private static double[] createRandomArray(int size, long seed) {
        Random random = new Random(seed);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = random.nextDouble() * 2.0 - 1.0;
        }
        return out;
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private static void writeCsv(
            double objectAvgMs,
            double byteBufferAvgMs,
            int objectPayloadBytes,
            int byteBufferPayloadBytes,
            double speedup) throws IOException {
        Path resultsDir = Path.of("results");
        Files.createDirectories(resultsDir);

        Path csvPath = resultsDir.resolve("serial_bench.csv");
        StringBuilder sb = new StringBuilder();
        sb.append("method,avg_ms,payload_bytes,array_size,rounds").append(System.lineSeparator());
        sb.append(String.format(Locale.ROOT,
                "ObjectOutputStream,%.6f,%d,%d,%d%n",
                objectAvgMs, objectPayloadBytes, ARRAY_SIZE, MEASURE_ROUNDS));
        sb.append(String.format(Locale.ROOT,
                "ByteBuffer,%.6f,%d,%d,%d%n",
                byteBufferAvgMs, byteBufferPayloadBytes, ARRAY_SIZE, MEASURE_ROUNDS));
        sb.append(String.format(Locale.ROOT,
                "speedup_object_over_bytebuffer,%.6f,%d,%d,%d%n",
                speedup, -1, ARRAY_SIZE, MEASURE_ROUNDS));

        Files.writeString(
                csvPath,
                sb.toString(),
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
    }
}
