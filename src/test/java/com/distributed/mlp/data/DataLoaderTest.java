package com.distributed.mlp.data;

import com.distributed.mlp.data.DataLoader.Sample;
import java.nio.file.Path;
import java.util.List;

/**
 * Lightweight DataLoader checks for shard sizing and pixel normalization bounds.
 */
public final class DataLoaderTest {
    private DataLoaderTest() {
    }

    public static void main(String[] args) throws Exception {
        DataLoader loader = new DataLoader();
        Path datasetRoot = Path.of("data", "food-101", "images");

        // Use many virtual workers so this smoke test stays memory-safe.
        int totalWorkers = 1000;
        List<Sample> shard = loader.loadShard(datasetRoot, 0, totalWorkers, 42L);

        assertTrue(!shard.isEmpty(), "Shard must not be empty");
        assertTrue(shard.size() <= 200, "Shard size should be small for 1000-way split");

        int inspected = Math.min(5, shard.size());
        for (int i = 0; i < inspected; i++) {
            Sample sample = shard.get(i);
            double[] pixels = sample.pixels();

            assertTrue(pixels.length == DataLoader.INPUT_SIZE, "Unexpected input vector size");
            for (double v : pixels) {
                assertTrue(v >= 0.0 && v <= 1.0, "Pixel out of [0,1] range: " + v);
            }
            assertTrue(sample.label() >= 0, "Label must be non-negative");
        }

        System.out.println("DataLoaderTest PASS: shard_count=" + shard.size() + ", inspected=" + inspected);
    }

    private static void assertTrue(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}
