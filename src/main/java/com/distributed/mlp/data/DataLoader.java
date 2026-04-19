package com.distributed.mlp.data;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import javax.imageio.ImageIO;

/**
 * Loads Food-101 images, normalizes pixels to [0,1], and creates deterministic worker shards.
 */
public class DataLoader {
    public static final int TARGET_WIDTH = 128;
    public static final int TARGET_HEIGHT = 128;
    public static final int CHANNELS = 3;
    public static final int INPUT_SIZE = TARGET_WIDTH * TARGET_HEIGHT * CHANNELS;
    public static final long DEFAULT_SEED = 42L;

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

    /**
     * Loads a worker shard from the default dataset path.
     */
    public List<Sample> loadShard(int workerId, int totalWorkers) throws IOException {
        Path defaultRoot = Path.of("data", "food-101", "images");
        return loadShard(defaultRoot, workerId, totalWorkers, DEFAULT_SEED);
    }

    /**
     * Loads a worker shard from the provided dataset root using deterministic shuffling.
     */
    public List<Sample> loadShard(Path datasetRoot, int workerId, int totalWorkers, long seed) throws IOException {
        if (datasetRoot == null) {
            throw new IllegalArgumentException("datasetRoot must not be null");
        }
        if (workerId < 0 || totalWorkers <= 0 || workerId >= totalWorkers) {
            throw new IllegalArgumentException("Invalid shard args: workerId=" + workerId + ", totalWorkers=" + totalWorkers);
        }
        if (!Files.isDirectory(datasetRoot)) {
            throw new IOException("Dataset root not found: " + datasetRoot.toAbsolutePath());
        }

        List<Path> imagePaths = discoverImages(datasetRoot);
        if (imagePaths.isEmpty()) {
            return List.of();
        }

        Collections.shuffle(imagePaths, new java.util.Random(seed));
        Map<String, Integer> labelMap = buildLabelMap(imagePaths);

        List<Sample> shard = new ArrayList<>();
        for (int i = 0; i < imagePaths.size(); i++) {
            if (i % totalWorkers != workerId) {
                continue;
            }
            Path imagePath = imagePaths.get(i);
            String className = imagePath.getParent().getFileName().toString();
            Integer label = labelMap.get(className);
            if (label == null) {
                throw new IOException("Missing label for class: " + className);
            }
            double[] pixels = loadAndNormalize(imagePath);
            shard.add(new Sample(pixels, label, imagePath));
        }
        return shard;
    }

    private static List<Path> discoverImages(Path datasetRoot) throws IOException {
        List<Path> imagePaths = new ArrayList<>();
        try (var stream = Files.walk(datasetRoot)) {
            stream
                .filter(Files::isRegularFile)
                .filter(DataLoader::isSupportedImage)
                .forEach(imagePaths::add);
        }
        imagePaths.sort(Comparator.comparing(Path::toString));
        return imagePaths;
    }

    private static boolean isSupportedImage(Path path) {
        String fileName = path.getFileName().toString();
        int dot = fileName.lastIndexOf('.');
        if (dot <= 0 || dot == fileName.length() - 1) {
            return false;
        }
        String ext = fileName.substring(dot + 1).toLowerCase(Locale.ROOT);
        return SUPPORTED_EXTENSIONS.contains(ext);
    }

    private static Map<String, Integer> buildLabelMap(List<Path> imagePaths) {
        Set<String> labels = new HashSet<>();
        for (Path p : imagePaths) {
            labels.add(p.getParent().getFileName().toString());
        }
        List<String> sortedLabels = new ArrayList<>(labels);
        sortedLabels.sort(String::compareTo);

        Map<String, Integer> labelMap = new HashMap<>();
        for (int i = 0; i < sortedLabels.size(); i++) {
            labelMap.put(sortedLabels.get(i), i);
        }
        return labelMap;
    }

    private static double[] loadAndNormalize(Path imagePath) throws IOException {
        BufferedImage original = ImageIO.read(imagePath.toFile());
        if (original == null) {
            throw new IOException("Unsupported image format: " + imagePath);
        }

        BufferedImage resized = new BufferedImage(TARGET_WIDTH, TARGET_HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g.drawImage(original, 0, 0, TARGET_WIDTH, TARGET_HEIGHT, null);
        } finally {
            g.dispose();
        }

        double[] pixels = new double[INPUT_SIZE];
        int idx = 0;
        for (int y = 0; y < TARGET_HEIGHT; y++) {
            for (int x = 0; x < TARGET_WIDTH; x++) {
                int rgb = resized.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                pixels[idx++] = r / 255.0;
                pixels[idx++] = green / 255.0;
                pixels[idx++] = b / 255.0;
            }
        }
        return pixels;
    }
}
