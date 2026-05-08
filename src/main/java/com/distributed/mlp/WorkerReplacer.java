package com.distributed.mlp;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Watches for dead worker IDs and spawns fresh JVM replacements for each one.
 * Only activated during crash tests via system property: mlp.enableReplacer=true
 */
public final class WorkerReplacer {

    private static final int  MAX_REPLACEMENTS = 3;
    private static final long REPLACE_DELAY_MS = 2_000;

    private final String masterHost;
    private final int    masterPort;
    private final int    totalWorkers;
    private final int    stepsPerWorker;
    private final long   baseSeed;

    /** workerId → how many replacements have already been spawned for it. */
    private final Map<Integer, Integer> replacementCounts = new ConcurrentHashMap<>();

    /** Dead worker IDs queued for replacement. */
    private final BlockingQueue<Integer> deadQueue = new LinkedBlockingQueue<>();

    /** Live replacement Process handles so we can kill them on shutdown. */
    private final Map<Integer, Process> liveProcesses = new ConcurrentHashMap<>();

    /**
     * Queue of worker IDs that WorkerReplacer has decided to spawn, in the
     * order they will connect to the Master.  The Master's background accept
     * loop calls {@link #nextExpectedId()} to learn which ID to assign to each
     * incoming replacement socket — without any extra bytes on the wire.
     */
    private final BlockingQueue<Integer> expectedIds = new LinkedBlockingQueue<>();

    private volatile boolean running = false;
    private Thread replacerThread;
    private final boolean enabled;

    /**
     * Create WorkerReplacer. It only activates if system property 
     * "mlp.enableReplacer" is set to "true".
     */
    public WorkerReplacer(String masterHost, int masterPort,
                          int totalWorkers, int stepsPerWorker, long baseSeed) {
        this.masterHost     = masterHost;
        this.masterPort     = masterPort;
        this.totalWorkers   = totalWorkers;
        this.stepsPerWorker = stepsPerWorker;
        this.baseSeed       = baseSeed;
        this.enabled = Boolean.parseBoolean(
            System.getProperty("mlp.enableReplacer", "false"));
        
        if (enabled) {
            System.out.println("[WorkerReplacer] Worker replacement is ENABLED");
        } else {
            System.out.println("[WorkerReplacer] Worker replacement is DISABLED " +
                "(set -Dmlp.enableReplacer=true to enable)");
        }
    }

    /** 
     * Static helper to check if replacer should be created.
     * Use this in Master to decide whether to instantiate WorkerReplacer.
     */
    public static boolean isEnabled() {
        return Boolean.parseBoolean(
            System.getProperty("mlp.enableReplacer", "false"));
    }

    /** Start the background replacer daemon thread (only if enabled). */
    public void start() {
        if (!enabled) {
            System.out.println("[WorkerReplacer] Not starting — disabled");
            return;
        }
        running = true;
        replacerThread = new Thread(this::replacerLoop, "worker-replacer");
        replacerThread.setDaemon(true);
        replacerThread.start();
        System.out.println("[WorkerReplacer] Started.");
    }

    /** Stop the replacer and kill any processes it spawned. */
    public void stop() {
        running = false;
        if (replacerThread != null) replacerThread.interrupt();
        for (Map.Entry<Integer, Process> e : liveProcesses.entrySet()) {
            Process p = e.getValue();
            if (p != null && p.isAlive()) {
                p.destroyForcibly();
                System.out.printf("[WorkerReplacer] Killed replacement process for worker %d.%n",
                        e.getKey());
            }
        }
        liveProcesses.clear();
    }

    /**
     * Called by Master's WorkerHandler when a worker crashes or disconnects
     * unexpectedly. Safe to call from any thread.
     * Silently ignores if disabled.
     */
    public void reportDead(int workerId) {
        if (!enabled) {
            System.out.printf("[WorkerReplacer] Worker %d reported dead but replacer disabled — ignoring%n", 
                workerId);
            return;
        }
        
        int count = replacementCounts.getOrDefault(workerId, 0);
        if (count >= MAX_REPLACEMENTS) {
            System.err.printf("[WorkerReplacer] Worker %d has been replaced %d/%d times — giving up.%n",
                    workerId, count, MAX_REPLACEMENTS);
            return;
        }
        System.out.printf("[WorkerReplacer] Worker %d reported dead — queuing replacement "
                + "(attempt %d/%d).%n", workerId, count + 1, MAX_REPLACEMENTS);
        deadQueue.offer(workerId);
    }

    /**
     * Blocks until WorkerReplacer has decided which worker ID the next
     * incoming replacement connection belongs to, then returns that ID.
     * Called by Master's background accept loop for each replacement socket.
     * Returns -1 if replacer is disabled.
     */
    public int nextExpectedId() throws IOException {
        if (!enabled) {
            throw new IOException("WorkerReplacer is disabled — unexpected replacement connection");
        }
        try {
            return expectedIds.take();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted waiting for replacement worker ID", e);
        }
    }
    
    /** Check if there are pending expected replacement connections. */
    public boolean hasPendingReplacements() {
        return enabled && !expectedIds.isEmpty();
    }
    
    /** Check if replacer is enabled and running. */
    public boolean isRunning() {
        return enabled && running;
    }

    // ── private ───────────────────────────────────────────────────────────

    private void replacerLoop() {
        while (running) {
            try {
                int workerId = deadQueue.take();

                System.out.printf("[WorkerReplacer] Waiting %dms before spawning replacement "
                        + "for worker %d...%n", REPLACE_DELAY_MS, workerId);
                Thread.sleep(REPLACE_DELAY_MS);

                if (!running) break;
                spawnReplacement(workerId);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        System.out.println("[WorkerReplacer] Loop exiting.");
    }

    /**
     * Forks a new {@link Worker} JVM with the same arguments as the original.
     * The replacement connects to masterPort just like an original worker — no
     * protocol change on either side.
     */
    private void spawnReplacement(int workerId) {
        int attempt = replacementCounts.merge(workerId, 1, Integer::sum);
        
        // Double-check max replacements
        if (attempt > MAX_REPLACEMENTS) {
            System.err.printf("[WorkerReplacer] Worker %d exceeded max replacements (%d)%n", 
                workerId, MAX_REPLACEMENTS);
            replacementCounts.put(workerId, MAX_REPLACEMENTS); // cap it
            return;
        }

        // Worker seed matches the formula in Worker.run(): seed + workerId
        long workerSeed = baseSeed + workerId;

        List<String> cmd = new ArrayList<>(List.of(
                "java",
                "-Xmx512m",
                "-XX:+ExitOnOutOfMemoryError",
                "-cp", currentClasspath(),
                "-Dmlp.enableReplacer=true",  // Pass through to nested workers
                "-Dmlp.logFile=logs/worker-" + workerId + "-replacement-" + attempt + ".log",
                "com.distributed.mlp.Worker",
                masterHost,
                String.valueOf(masterPort),
                String.valueOf(workerId),       // same shard as the dead worker
                String.valueOf(totalWorkers),
                String.valueOf(stepsPerWorker),
                String.valueOf(workerSeed)
        ));

        System.out.printf("[WorkerReplacer] Spawning replacement for worker %d "
                + "(attempt %d/%d): %s%n", workerId, attempt, MAX_REPLACEMENTS, 
                String.join(" ", cmd));

        // Publish the ID BEFORE spawning so the Master's accept loop
        // can call nextExpectedId() as soon as the socket arrives.
        expectedIds.offer(workerId);

        try {
            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            
            java.io.File logFile = logFile(workerId, attempt);
            logFile.getParentFile().mkdirs();
            pb.redirectOutput(ProcessBuilder.Redirect.appendTo(logFile));

            Process proc = pb.start();
            liveProcesses.put(workerId, proc);
            
            System.out.printf("[WorkerReplacer] Replacement process started for worker %d (PID: %d)%n", 
                workerId, proc.pid());

            int finalAttempt = attempt;
            Thread watcher = new Thread(
                    () -> watchProcess(proc, workerId, finalAttempt),
                    "replacement-watcher-" + workerId + "-" + attempt);
            watcher.setDaemon(true);
            watcher.start();

        } catch (IOException e) {
            System.err.printf("[WorkerReplacer] Failed to spawn replacement for worker %d: %s%n",
                    workerId, e.getMessage());
            e.printStackTrace();
            // Remove the expected ID since spawning failed
            expectedIds.remove(workerId);
        }
    }

    private void watchProcess(Process proc, int workerId, int attempt) {
        try {
            int exitCode = proc.waitFor();
            liveProcesses.remove(workerId);
            if (exitCode == 0) {
                System.out.printf("[WorkerReplacer] Replacement worker %d (attempt %d) finished cleanly.%n",
                        workerId, attempt);
            } else {
                System.err.printf("[WorkerReplacer] Replacement worker %d (attempt %d) exited with code %d "
                        + "— re-reporting as dead.%n", workerId, attempt, exitCode);
                Thread.sleep(1000); // Brief delay before re-reporting
                reportDead(workerId);   // will respect MAX_REPLACEMENTS
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static String currentClasspath() {
        String cp = System.getProperty("java.class.path");
        return (cp != null && !cp.isBlank()) ? cp : "target/classes";
    }

    private static java.io.File logFile(int workerId, int attempt) {
        java.io.File dir = new java.io.File("logs");
        dir.mkdirs();
        return new java.io.File(dir, "worker_replacement_" + workerId + "_" + attempt + ".log");
    }
}