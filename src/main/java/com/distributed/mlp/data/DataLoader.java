package com.distributed.mlp.data;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * Loads Food-101 images, normalizes pixels to [0,1], and creates deterministic
 * worker shards.
 */
public class DataLoader {

    public static final int TARGET_WIDTH = 32;
    public static final int TARGET_HEIGHT = 32;
    public static final int INPUT_SIZE = 3072;
    public static final int CHANNELS = 3;
    public static final long DEFAULT_SEED = 42L;
    private static final int MAX_IMAGES = 100;

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of("jpg", "jpeg", "png");

    /**
     * Container for one training sample.
     */
    public static final class Sample {

        private final double[] pixels;
        private final int label;
        private final Path path;

        public Sample(double[] pixels, int label, Path path) {
            this.pixels = pixels;
            this.label = label;
            this.path = path;
        }

        public double[] pixels() {
            return pixels;
        }

        public int label() {
            return label;
        }

        public Path path() {
            return path;
        }
    }

    public List<Sample> loadShard(int workerId, int totalWorkers) throws IOException {
        return loadShard(workerId, totalWorkers, DEFAULT_SEED);
    }

    public List<Sample> loadShard(int workerId, int totalWorkers, long seed) throws IOException {
        if (workerId < 0 || totalWorkers <= 0 || workerId >= totalWorkers) {
            throw new IllegalArgumentException("Invalid shard args: workerId=" + workerId + ", totalWorkers=" + totalWorkers);
        }

        Path dataDir = Path.of("data/cifar-10-batches-bin");
        if (!Files.isDirectory(dataDir)) {
            throw new IOException("CIFAR-10 binary data not found at: " + dataDir.toAbsolutePath());
        }

        // First pass: count total samples
        int totalSamples = 0;
        for (int batch = 1; batch <= 5; batch++) {
            Path batchFile = dataDir.resolve("data_batch_" + batch + ".bin");
            totalSamples += (int) (Files.size(batchFile) / 3073);
        }

        // Build deterministic index list and shuffle, then pick this worker's indices
        List<Integer> indices = new ArrayList<>(totalSamples);
        for (int i = 0; i < totalSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(seed));

        // Only keep this worker's indices, sort them for sequential file access
        java.util.Set<Integer> myIndices = new java.util.HashSet<>();
        for (int i = workerId; i < totalSamples; i += totalWorkers) {
            myIndices.add(indices.get(i));
        }

        // Second pass: read only needed samples
        List<Sample> shard = new ArrayList<>(myIndices.size());
        int globalIdx = 0;
        for (int batch = 1; batch <= 5; batch++) {
            Path batchFile = dataDir.resolve("data_batch_" + batch + ".bin");
            try (DataInputStream dis = new DataInputStream(
                    new BufferedInputStream(Files.newInputStream(batchFile)))) {
                while (dis.available() >= 3073) {
                    int label = dis.readUnsignedByte();
                    if (myIndices.contains(globalIdx)) {
                        double[] pixels = new double[3072];
                        for (int i = 0; i < 3072; i++) {
                            pixels[i] = dis.readUnsignedByte() / 255.0;
                        }
                        shard.add(new Sample(pixels, label, null));
                    } else {
                        dis.skipBytes(3072);
                    }
                    globalIdx++;
                }
            }
        }
        return shard;
    }
}
