package com.distributed.mlp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.distributed.mlp.protocol.WeightSerializer;

/**
 * Checkpoint manager for the Master's global weight vector.
 *
 * Checkpoints are written atomically (write to .tmp, rename to final)
 * so a crash mid-write never leaves a corrupt file.
 *
 * File naming: results/checkpoint_<updateCount>.bin
 * The latest checkpoint can be found with {@link #findLatest()}.
 *
 * On startup, Master calls {@link #findLatest()} — if a file is returned
 * the weights are restored from it and training resumes from that update count.
 */
public final class Checkpoint {

    private static final Path RESULTS_DIR     = Path.of("results");
    private static final String PREFIX        = "checkpoint_";
    private static final String SUFFIX        = ".bin";
    private static final int    MAX_TO_KEEP   = 3;   // keep only the 3 most recent

    private Checkpoint() {}

    /**
     * Saves weights atomically.  The file is first written as a .tmp file
     * and then renamed so readers never see a partial write.
     *
     * @param weights     the global weight vector
     * @param updateCount current number of applied gradient updates (used in filename)
     */
    public static void save(double[] weights, int updateCount) throws IOException {
        Files.createDirectories(RESULTS_DIR);

        String name    = PREFIX + updateCount + SUFFIX;
        Path   tmp     = RESULTS_DIR.resolve(name + ".tmp");
        Path   final_  = RESULTS_DIR.resolve(name);

        byte[] bytes = WeightSerializer.toBytesDouble(weights);
        Files.write(tmp, bytes);
        Files.move(tmp, final_, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);

        System.out.printf("[Checkpoint] Saved %s (%d bytes, update #%d)%n",
                final_, bytes.length, updateCount);

        pruneOld();
    }

    /**
     * Returns the path of the most recent checkpoint, or null if none exist.
     */
    public static Path findLatest() throws IOException {
        if (!Files.isDirectory(RESULTS_DIR)) return null;

        try (Stream<Path> stream = Files.list(RESULTS_DIR)) {
            List<Path> checkpoints = stream
                    .filter(p -> p.getFileName().toString().startsWith(PREFIX)
                              && p.getFileName().toString().endsWith(SUFFIX))
                    .sorted(Comparator.comparingInt(Checkpoint::extractUpdate).reversed())
                    .collect(Collectors.toList());

            return checkpoints.isEmpty() ? null : checkpoints.get(0);
        }
    }

    /**
     * Returns the update count embedded in a checkpoint filename, or -1 on parse error.
     */
    public static int extractUpdate(Path p) {
        String name = p.getFileName().toString();
        try {
            String num = name.substring(PREFIX.length(), name.length() - SUFFIX.length());
            return Integer.parseInt(num);
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    /**
     * Loads weights from a checkpoint file.
     */
    public static double[] load(Path checkpointPath) throws IOException {
        byte[] bytes = Files.readAllBytes(checkpointPath);
        System.out.printf("[Checkpoint] Restored from %s (%d bytes)%n",
                checkpointPath, bytes.length);
        return WeightSerializer.fromBytesDouble(bytes);
    }

    // ------------------------------------------------------------------ //
    //  Private helpers                                                     //
    // ------------------------------------------------------------------ //

    /** Deletes all but the MAX_TO_KEEP most recent checkpoints. */
    private static void pruneOld() throws IOException {
        try (Stream<Path> stream = Files.list(RESULTS_DIR)) {
            List<Path> all = stream
                    .filter(p -> p.getFileName().toString().startsWith(PREFIX)
                              && p.getFileName().toString().endsWith(SUFFIX))
                    .sorted(Comparator.comparingInt(Checkpoint::extractUpdate).reversed())
                    .collect(Collectors.toList());

            for (int i = MAX_TO_KEEP; i < all.size(); i++) {
                Files.deleteIfExists(all.get(i));
                System.out.printf("[Checkpoint] Pruned old checkpoint: %s%n", all.get(i));
            }
        }
    }
}