package com.distributed.mlp;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

import com.distributed.mlp.protocol.MessageProtocol;
import com.distributed.mlp.protocol.MessageProtocol.ProtocolException;
import com.distributed.mlp.protocol.WeightSerializer;

public final class Master {

    private static final int    DEFAULT_PORT           = 9000;
    private static final int    DEFAULT_WORKERS        = 3;
    private static final int    DEFAULT_TARGET_UPDATES = Integer.MAX_VALUE;
    private static final double DEFAULT_LEARNING_RATE  = 1e-3;
    private static final int    CHECKPOINT_EVERY       = 5;
    private static final int    DEFAULT_STEPS          = 5;
    private static final long   DEFAULT_SEED           = 42L;

    private static final int TOTAL_PARAMETERS =
            com.distributed.mlp.model.MLPModel.HIDDEN1_DIM * com.distributed.mlp.model.MLPModel.INPUT_DIM
          + com.distributed.mlp.model.MLPModel.HIDDEN1_DIM
          + com.distributed.mlp.model.MLPModel.HIDDEN2_DIM * com.distributed.mlp.model.MLPModel.HIDDEN1_DIM
          + com.distributed.mlp.model.MLPModel.HIDDEN2_DIM
          + com.distributed.mlp.model.MLPModel.OUTPUT_DIM  * com.distributed.mlp.model.MLPModel.HIDDEN2_DIM
          + com.distributed.mlp.model.MLPModel.OUTPUT_DIM;

    private final int    port;
    private final int    expectedWorkers;
    private final double learningRate;
    private final double[] globalWeights;
    private final int    targetUpdates;
    private final int    stepsPerWorker;
    private final long   baseSeed;
    private final boolean enableCheckpoints;

    private final AtomicInteger  totalUpdates      = new AtomicInteger(0);
    private final AtomicBoolean  shutdownInitiated = new AtomicBoolean(false);
    private final ReentrantLock  weightLock        = new ReentrantLock();

    // Guarded by synchronized(workerChannels)
    private final List<WorkerChannel> workerChannels = new ArrayList<>();

    // All handler threads (initial + replacements), guarded by synchronized(workerThreads)
    private final List<Thread> workerThreads = new ArrayList<>();

    private final AtomicInteger liveWorkers;
    private final WorkerReplacer workerReplacer;

    // ── constructors ──────────────────────────────────────────────────────

    public Master(int port, int expectedWorkers) {
        this(port, expectedWorkers, DEFAULT_STEPS, DEFAULT_SEED, true);
    }

    public Master(int port, int expectedWorkers, int stepsPerWorker, long baseSeed) {
        this(port, expectedWorkers, stepsPerWorker, baseSeed, true);
    }

    public Master(int port, int expectedWorkers, int stepsPerWorker, long baseSeed, boolean enableCheckpoints) {
        this.port            = port;
        this.expectedWorkers = expectedWorkers;
        this.learningRate    = DEFAULT_LEARNING_RATE;
        this.globalWeights   = new double[TOTAL_PARAMETERS];
        this.targetUpdates   = expectedWorkers * stepsPerWorker;
        this.stepsPerWorker  = stepsPerWorker;
        this.baseSeed        = baseSeed;
        this.enableCheckpoints = enableCheckpoints;
        this.liveWorkers     = new AtomicInteger(0); // incremented as handlers register
        this.workerReplacer  = new WorkerReplacer(
                "127.0.0.1", port, expectedWorkers, stepsPerWorker, baseSeed);
    }

    public static void main(String[] args) {
        int  port    = args.length >= 1 ? Integer.parseInt(args[0]) : DEFAULT_PORT;
        int  workers = args.length >= 2 ? Integer.parseInt(args[1]) : DEFAULT_WORKERS;
        int  steps   = args.length >= 3 ? Integer.parseInt(args[2]) : DEFAULT_STEPS;
        long seed    = args.length >= 4 ? Long.parseLong(args[3])   : DEFAULT_SEED;
        boolean enableCheckpoints = args.length < 5 || Boolean.parseBoolean(args[4]);

        Master master = new Master(port, workers, steps, seed, enableCheckpoints);
        try {
            master.start();
        } catch (IOException e) {
            System.err.println("Master failed: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    // ── start ─────────────────────────────────────────────────────────────

    public void start() throws IOException {
        if (enableCheckpoints) {
            // Restore from checkpoint if one exists
            try {
                Path latest = Checkpoint.findLatest();
                if (latest != null) {
                    double[] restored = Checkpoint.load(latest);
                    if (restored.length == globalWeights.length) {
                        System.arraycopy(restored, 0, globalWeights, 0, restored.length);
                        int resumeAt = Checkpoint.extractUpdate(latest);
                        totalUpdates.set(resumeAt);
                        System.out.printf("[Master] Resumed from checkpoint at update #%d%n", resumeAt);
                    } else {
                        System.err.println("[Master] Checkpoint size mismatch — starting fresh.");
                    }
                } else {
                    System.out.println("[Master] No checkpoint found — starting fresh.");
                }
            } catch (IOException e) {
                System.err.println("[Master] Could not read checkpoint: " + e.getMessage());
            }
        } else {
            System.out.println("[Master] Checkpointing disabled.");
        }

        workerReplacer.start();

        // KEY FIX: keep the ServerSocket open for the entire training run.
        // We do NOT close it after accepting initial workers — the background
        // accept loop below uses it to admit replacement workers at any time.
        // Worker.java needs ZERO changes: replacements connect on the same port
        // with the same protocol as the originals.
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            serverSocket.setReuseAddress(true);
            System.out.printf("Master listening on port %d, waiting for %d workers...%n",
                    port, expectedWorkers);

            // ── Accept initial workers (blocking, in order) ──────────────
            for (int id = 0; id < expectedWorkers; id++) {
                Socket s = serverSocket.accept();
                System.out.printf("Worker %d connected from %s%n", id, s.getRemoteSocketAddress());
                startHandler(id, s);
            }
            System.out.printf("All %d initial workers connected.%n", expectedWorkers);

            // ── Background accept loop for replacement workers ───────────
            // Replacement workers connect with the same Worker binary and the
            // same args — there is no extra handshake byte.  WorkerReplacer
            // ensures the correct workerId is passed as a JVM argument, so the
            // replacement inherits the same data shard deterministically.
            //
            // The Master does NOT know the replacement's worker ID from the
            // connection alone — it learns it via the first PULL_REQUEST or
            // PUSH_GRADIENT, but the WorkerHandler only needs the ID for
            // logging and channel management.  WorkerReplacer already chose
            // the ID when it spawned the process, so we assign a synthetic
            // "reconnect ID" by reading the next slot from a counter that
            // WorkerReplacer updates.  In practice the simplest correct approach
            // is: accept the socket, start a new handler with the SAME workerId
            // that WorkerReplacer told us it spawned.  We do this by having
            // WorkerReplacer register the expected next ID in a shared queue.
            Thread acceptLoop = new Thread(() -> {
                while (!shutdownInitiated.get()) {
                    try {
                        // SO_TIMEOUT so the loop can notice shutdownInitiated
                        serverSocket.setSoTimeout(1000);
                        Socket s;
                        try {
                            s = serverSocket.accept();
                        } catch (java.net.SocketTimeoutException e) {
                            continue;  // check shutdownInitiated and loop
                        }

                        // Ask WorkerReplacer which ID it assigned to this connection
                        int replacementId = workerReplacer.nextExpectedId();
                        System.out.printf("[Master] Replacement worker connected: id=%d from %s%n",
                                replacementId, s.getRemoteSocketAddress());
                        startHandler(replacementId, s);

                    } catch (IOException e) {
                        if (!shutdownInitiated.get()) {
                            System.err.println("[Master] Accept loop error: " + e.getMessage());
                        }
                        break;
                    }
                }
                System.out.println("[Master] Accept loop exited.");
            }, "master-accept-loop");
            acceptLoop.setDaemon(true);
            acceptLoop.start();

            // Wait for initial handler threads (replacements are daemons and
            // are cleaned up when the JVM exits after training completes)
            List<Thread> initialThreads;
            synchronized (workerThreads) {
                initialThreads = new ArrayList<>(workerThreads);
            }
            for (Thread t : initialThreads) {
                try { t.join(); }
                catch (InterruptedException e) { Thread.currentThread().interrupt(); break; }
            }
            // ServerSocket closed by try-with-resources → unblocks accept loop
        }

        workerReplacer.stop();
        System.out.println("[Master] Shutdown complete.");
    }

    // ── package helpers ───────────────────────────────────────────────────

    private void startHandler(int workerId, Socket s) {
        Thread handler = new Thread(new WorkerHandler(workerId, s),
                "master-worker-handler-" + workerId);
        handler.setDaemon(true);
        handler.start();
        synchronized (workerThreads) { workerThreads.add(handler); }
    }

    public double[] snapshotGlobalWeights() {
        weightLock.lock();
        try { return Arrays.copyOf(globalWeights, globalWeights.length); }
        finally { weightLock.unlock(); }
    }

    public int applyGradient(double[] gradient) {
        if (gradient.length != globalWeights.length)
            throw new IllegalArgumentException("Gradient length mismatch");
        weightLock.lock();
        try {
            for (int i = 0; i < globalWeights.length; i++)
                globalWeights[i] -= learningRate * gradient[i];
        } finally { weightLock.unlock(); }

        int n = totalUpdates.incrementAndGet();

        if (enableCheckpoints && n % CHECKPOINT_EVERY == 0) {
            double[] snap = snapshotGlobalWeights();
            new Thread(() -> {
                try { Checkpoint.save(snap, n); }
                catch (IOException e) {
                    System.err.println("[Master] Checkpoint save failed: " + e.getMessage());
                }
            }, "checkpoint-writer-" + n).start();
        }

        return n;
    }

    private void registerChannel(WorkerChannel ch) {
        synchronized (workerChannels) { workerChannels.add(ch); }
        liveWorkers.incrementAndGet();
    }

    private void unregisterChannel(WorkerChannel ch) {
        boolean removed;
        synchronized (workerChannels) { removed = workerChannels.remove(ch); }
        if (!removed) return;  // already removed (e.g. watchdog + EOF both fire)

        int remaining = liveWorkers.decrementAndGet();
        System.out.printf("[Master] Worker %d unregistered. Live workers: %d%n",
                ch.workerId, remaining);

        if (enableCheckpoints && remaining == 0 && !shutdownInitiated.get()) {
            System.err.println("[Master] All workers disconnected unexpectedly — saving emergency checkpoint.");
            try { Checkpoint.save(snapshotGlobalWeights(), totalUpdates.get()); }
            catch (IOException e) {
                System.err.println("[Master] Emergency checkpoint failed: " + e.getMessage());
            }
        }
    }

    private void deleteAllCheckpoints() {
        try {
            java.nio.file.Path dir = java.nio.file.Path.of("results");
            if (!java.nio.file.Files.exists(dir)) return;
            try (var stream = java.nio.file.Files.newDirectoryStream(dir, "checkpoint_*.bin")) {
                for (var p : stream) {
                    java.nio.file.Files.deleteIfExists(p);
                }
            }
            System.out.println("[Master] All checkpoints deleted.");
        } catch (IOException e) {
            System.err.println("[Master] Could not delete checkpoints: " + e.getMessage());
        }
    }

    private void broadcastShutdown() {
        saveWeights();
        if (enableCheckpoints) {
            deleteAllCheckpoints();
            try { Checkpoint.save(snapshotGlobalWeights(), totalUpdates.get()); }
            catch (IOException e) {
                System.err.println("[Master] Final checkpoint failed: " + e.getMessage());
            }
        }

        List<WorkerChannel> snapshot;
        synchronized (workerChannels) { snapshot = new ArrayList<>(workerChannels); }
        for (WorkerChannel ch : snapshot) {
            try { ch.sendShutdown(); }
            catch (IOException | ProtocolException e) {
                System.err.printf("[Master] SHUTDOWN to worker %d failed: %s%n",
                        ch.workerId, e.getMessage());
            }
        }
    }

    private void saveWeights() {
        try {
            java.nio.file.Path dir = java.nio.file.Path.of("results");
            java.nio.file.Files.createDirectories(dir);
            int num = 0;
            while (java.nio.file.Files.exists(dir.resolve("model_weights_" + num + ".bin"))) num++;
            java.nio.file.Path out = dir.resolve("model_weights_" + num + ".bin");
            java.nio.file.Files.write(out, WeightSerializer.toBytesDouble(globalWeights));
            System.out.printf("[Master] Weights saved to %s%n", out);
        } catch (IOException e) {
            System.err.println("[Master] saveWeights failed: " + e.getMessage());
        }
    }

    // ================================================================== //
    //  Inner classes                                                       //
    // ================================================================== //

    private final class WorkerHandler implements Runnable {
        private final int    workerId;
        private final Socket socket;
        private volatile long lastGradientPushMs = System.currentTimeMillis();
        private static final long GRADIENT_TIMEOUT_MS = 60_000; 

        WorkerHandler(int workerId, Socket socket) {
            this.workerId = workerId;
            this.socket   = socket;
        }

        @Override
        public void run() {
            try (Socket s = socket;
                 DataInputStream  in  = new DataInputStream(s.getInputStream());
                 DataOutputStream out = new DataOutputStream(s.getOutputStream())) {

                WorkerChannel channel = new WorkerChannel(workerId, out);
                registerChannel(channel);  // also increments liveWorkers

                Thread watchdog = new Thread(() -> {
                    while (!shutdownInitiated.get() && !Thread.currentThread().isInterrupted()) {
                        try { Thread.sleep(5_000); } catch (InterruptedException e) { break; }
                        long silent = System.currentTimeMillis() - lastGradientPushMs;
                        if (silent > GRADIENT_TIMEOUT_MS) {
                            System.err.printf("[Master] Worker %d silent for %ds — declaring dead.%n",
                                    workerId, silent / 1000);
                            unregisterChannel(channel);
                            workerReplacer.reportDead(workerId);
                            try { socket.close(); } catch (IOException ignored) {}
                            break;
                        }
                    }
                }, "gradient-watchdog-" + workerId);
                watchdog.setDaemon(true);
                watchdog.start();

                try {
                    while (true) {
                        if (shutdownInitiated.get()) return;

                        int[] header     = MessageProtocol.readHeader(in);
                        int   totalLen   = header[0];
                        int   type       = header[1];
                        int   payloadLen = totalLen - MessageProtocol.HEADER_SIZE;

                        switch (type) {
                            case MessageProtocol.PULL_REQUEST -> {
                                if (payloadLen > 0) in.skipNBytes(payloadLen);
                                handlePullRequest(channel);
                            }
                            case MessageProtocol.PUSH_GRADIENT ->
                                    handlePushGradient(in, payloadLen);

                            default -> {
                                if (payloadLen > 0) in.skipNBytes(payloadLen);
                                System.err.printf("[Master] Worker %d unknown type 0x%02X%n",
                                        workerId, type);
                            }
                        }
                    }
                } finally {
                    unregisterChannel(channel);
                }

            } catch (EOFException | SocketException e) {
                System.out.printf("[Master] Worker %d disconnected.%n", workerId);
                if (!shutdownInitiated.get()) {
                    System.out.printf("[Master] Unexpected disconnect — requesting replacement "
                            + "for worker %d.%n", workerId);
                    workerReplacer.reportDead(workerId);
                }
            } catch (ProtocolException e) {
                System.err.printf("[Master] Worker %d protocol error: %s%n",
                        workerId, e.getMessage());
                if (!shutdownInitiated.get()) workerReplacer.reportDead(workerId);
            } catch (IOException e) {
                System.err.printf("[Master] Worker %d I/O error: %s%n",
                        workerId, e.getMessage());
                if (!shutdownInitiated.get()) workerReplacer.reportDead(workerId);
            }
        }

        private void handlePullRequest(WorkerChannel ch)
                throws IOException, ProtocolException {
            byte[] payload = WeightSerializer.toBytesDouble(snapshotGlobalWeights());
            ch.sendWeightResponse(payload);
            System.out.printf("[Master] Worker %d <- WEIGHT_RESPONSE (%d bytes)%n",
                    workerId, payload.length);
        }

       private void handlePushGradient(DataInputStream in, int payloadLen)
        throws IOException {
            if (payloadLen <= 0) {
                System.err.printf("[Master] Worker %d empty gradient push.%n", workerId);
                return;
            }
            byte[]   bytes    = new byte[payloadLen];
            in.readFully(bytes);
            double[] gradient = WeightSerializer.fromBytesDouble(bytes);
            lastGradientPushMs = System.currentTimeMillis(); // ADD THIS
            int      n        = applyGradient(gradient);
            System.out.printf("[Master] Worker %d -> gradient applied (update #%d)%n",
                    workerId, n);

            if (n >= targetUpdates && shutdownInitiated.compareAndSet(false, true)) {
                System.out.printf("[Master] TARGET_UPDATES=%d reached. Broadcasting SHUTDOWN.%n", n);
                broadcastShutdown();
            }
        }
    }

    private static final class WorkerChannel {
        final int workerId;
        private final DataOutputStream out;
        private final ReentrantLock    sendLock = new ReentrantLock();

        WorkerChannel(int workerId, DataOutputStream out) {
            this.workerId = workerId;
            this.out      = out;
        }

        void sendWeightResponse(byte[] payload) throws IOException, ProtocolException {
            sendLock.lock();
            try {
                MessageProtocol.writeHeader(out, MessageProtocol.WEIGHT_RESPONSE, payload.length);
                out.write(payload);
                out.flush();
            } finally { sendLock.unlock(); }
        }

        void sendShutdown() throws IOException, ProtocolException {
            sendLock.lock();
            try {
                MessageProtocol.writeHeader(out, MessageProtocol.SHUTDOWN, 0);
                out.flush();
            } finally { sendLock.unlock(); }
        }
    }
}