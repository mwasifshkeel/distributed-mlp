package com.distributed.mlp.gui;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;

import javafx.application.Platform;

/**
 * Manages all child JVM processes launched by the dashboard.
 * All processes share a single thread pool for I/O pumping.
 */
public class ProcessManager {

    private static final ProcessManager INSTANCE = new ProcessManager();
    public static ProcessManager getInstance() { return INSTANCE; }

    private final ExecutorService pool = Executors.newCachedThreadPool(r -> {
        Thread t = new Thread(r, "process-io");
        t.setDaemon(true);
        return t;
    });

    private final List<Process> running = Collections.synchronizedList(new ArrayList<>());

    // Detect project root: look for pom.xml walking up from working dir
    public static Path projectRoot() {
        Path p = Paths.get(System.getProperty("user.dir"));
        while (p != null) {
            if (p.resolve("pom.xml").toFile().exists()) return p;
            p = p.getParent();
        }
        return Paths.get(System.getProperty("user.dir"));
    }

    public static String classpath() {
        return projectRoot().resolve("target/classes").toString();
    }

    public static Path logsDir() {
        Path dir = projectRoot().resolve("logs").resolve("gui");
        try {
            Files.createDirectories(dir);
        } catch (Exception ignored) {
        }
        return dir;
    }

    /**
     * Launch a class with the given args. Streams stdout/stderr to the consumer on the FX thread.
     * Returns a Future<Integer> exit code.
     */
    public Future<Integer> launch(String mainClass, List<String> args,
                                   Consumer<String> lineConsumer,
                                   Consumer<Integer> onExit) {
        return pool.submit(() -> {
            return runProcess(
                startJavaProcess(mainClass, args, List.of(), null),
                mainClass,
                lineConsumer,
                onExit,
                ""
            );
        });
    }

    /** Launch a shell script with args and env vars. */
    public Future<Integer> launchScript(String scriptName, List<String> args, Map<String, String> env,
                                        Consumer<String> lineConsumer, Consumer<Integer> onExit) {
        return pool.submit(() -> {
            Path scriptPath = projectRoot().resolve(scriptName);
            List<String> cmd = new ArrayList<>();
            cmd.add("bash");
            cmd.add(scriptPath.toString());
            cmd.addAll(args);

                lineConsumer.accept(">> " + String.join(" ", cmd));

            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectErrorStream(true);
            pb.directory(projectRoot().toFile());
            if (env != null) {
                pb.environment().putAll(env);
            }
            Process proc = pb.start();
            running.add(proc);
            return runProcess(proc, scriptName, lineConsumer, onExit, "");
        });
    }

    /** Launch Master + N Workers for distributed training. */
    public void launchDistributed(int port, int workers, int steps, int seed,
                                   Consumer<String> log, Consumer<Boolean> onDone) {
        pool.submit(() -> {
            try {
                // Start master
                List<String> masterArgs = List.of(
                    String.valueOf(port),
                    String.valueOf(workers),
                    String.valueOf(steps),
                    String.valueOf(seed)
                );
                Process master = startJavaProcess(
                    "com.distributed.mlp.Master",
                    masterArgs,
                    List.of(),
                    "256m");
                pipeProcess(master, "Master", log);

                Thread.sleep(2000); // let master bind

                // Start workers
                List<Process> workerProcs = new ArrayList<>();
                for (int i = 0; i < workers; i++) {
                    List<String> wArgs = List.of(
                        "127.0.0.1",
                        String.valueOf(port),
                        String.valueOf(i),
                        String.valueOf(workers),
                        String.valueOf(steps),
                        String.valueOf(seed)
                    );
                        Process wp = startJavaProcess(
                            "com.distributed.mlp.Worker",
                            wArgs,
                            List.of(),
                            "512m");
                        workerProcs.add(wp);
                        pipeProcess(wp, "Worker-" + i, log);
                    Thread.sleep(200);
                }

                // Wait for all workers
                for (Process wp : workerProcs) wp.waitFor();
                master.waitFor();

                Platform.runLater(() -> {
                    log.accept("✅ Training complete.");
                    onDone.accept(true);
                });
            } catch (Exception ex) {
                Platform.runLater(() -> {
                    log.accept("❌ Error: " + ex.getMessage());
                    onDone.accept(false);
                });
            }
        });
    }

    private Process startJavaProcess(String mainClass, List<String> args, List<String> jvmProps, String heapMb)
            throws Exception {
        List<String> cmd = new ArrayList<>();
        cmd.add(javaExecutable());
        cmd.addAll(jvmFlags());
        if (jvmProps != null && !jvmProps.isEmpty()) {
            cmd.addAll(jvmProps);
        }
        if (heapMb != null && !heapMb.isBlank()) {
            cmd.add("-Xmx" + heapMb);
        }
        cmd.add("-cp");
        cmd.add(classpath());
        cmd.add(mainClass);
        cmd.addAll(args);

        ProcessBuilder pb = new ProcessBuilder(cmd);
        pb.redirectErrorStream(true);
        pb.directory(projectRoot().toFile());
        Process p = pb.start();
        running.add(p);
        return p;
    }

    /** Launch distributed training and kill one worker mid-way to test fault tolerance. */
    public void launchDistributedWithCrash(int port, int workers, int steps, int seed,
                                            int workerToCrash, int crashAfterSec,
                                            Consumer<String> log, Consumer<Boolean> onDone) {
        pool.submit(() -> {
            try {
                List<String> masterArgs = List.of(
                    String.valueOf(port), String.valueOf(workers),
                    String.valueOf(steps), String.valueOf(seed));
                Process master = startJavaProcess(
                    "com.distributed.mlp.Master",
                    masterArgs,
                    List.of(),
                    "256m");
                pipeProcess(master, "Master", log);
                Thread.sleep(2000);

                List<Process> workerProcs = new ArrayList<>();
                Process targetWorker = null;
                for (int i = 0; i < workers; i++) {
                    List<String> wArgs = List.of("127.0.0.1", String.valueOf(port),
                        String.valueOf(i), String.valueOf(workers),
                        String.valueOf(steps), String.valueOf(seed));
                        Process wp = startJavaProcess(
                            "com.distributed.mlp.Worker",
                            wArgs,
                            List.of(),
                            "512m");
                        pipeProcess(wp, "Worker-" + i, log);
                    workerProcs.add(wp);
                    if (i == workerToCrash) targetWorker = wp;
                    Thread.sleep(200);
                }

                if (targetWorker != null) {
                    final Process toKill = targetWorker;
                    Thread.sleep(crashAfterSec * 1000L);
                    Platform.runLater(() -> log.accept("💥 Killing worker " + workerToCrash + " (simulated crash)!"));
                    toKill.destroyForcibly();
                }

                for (Process wp : workerProcs) { if (wp.isAlive()) wp.waitFor(); }
                master.waitFor();

                Platform.runLater(() -> { log.accept("✅ Crash test session ended."); onDone.accept(true); });
            } catch (Exception ex) {
                Platform.runLater(() -> { log.accept("❌ Error: " + ex.getMessage()); onDone.accept(false); });
            }
        });
    }

    public void killAll() {
        synchronized (running) {
            for (Process p : running) {
                p.destroy();
            }
            running.clear();
        }
    }

    private static String javaExecutable() {
        return ProcessHandle.current().info().command().orElse("java");
    }

    private static List<String> jvmFlags() {
        return List.of();
    }

    private int runProcess(Process proc, String name,
                           Consumer<String> lineConsumer,
                           Consumer<Integer> onExit,
                           String linePrefix) throws Exception {
        Path logFile = createLogFile(name);
        int code = -1;
        try (BufferedReader br = new BufferedReader(new InputStreamReader(proc.getInputStream()));
             BufferedWriter writer = Files.newBufferedWriter(logFile, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                String prefixed = linePrefix.isBlank() ? line : linePrefix + line;
                writer.write(prefixed);
                writer.newLine();
                writer.flush();
                String out = prefixed;
                Platform.runLater(() -> lineConsumer.accept(out));
            }
            code = proc.waitFor();
        } finally {
            running.remove(proc);
            int finalCode = code;
            Platform.runLater(() -> onExit.accept(finalCode));
        }
        return code;
    }

    private void pipeProcess(Process proc, String name, Consumer<String> log) {
        pool.submit(() -> {
            try {
                runProcess(proc, name, log, code -> {
                }, "[" + name + "] ");
            } catch (Exception ignored) {
            }
        });
    }

    private static Path createLogFile(String name) {
        String shortName = name.contains(".")
                ? name.substring(name.lastIndexOf('.') + 1)
                : name;
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        return logsDir().resolve(shortName + "_" + timestamp + ".log");
    }
}