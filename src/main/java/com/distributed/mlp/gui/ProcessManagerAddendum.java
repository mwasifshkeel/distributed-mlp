package com.distributed.mlp.gui;
// ADDENDUM: Add this method to ProcessManager.java

// Paste this method inside the ProcessManager class body:

/*
    public void launchDistributedWithCrash(int port, int workers, int steps, int seed,
                                            int workerToCrash, int crashAfterSec,
                                            Consumer<String> log, Consumer<Boolean> onDone) {
        pool.submit(() -> {
            try {
                // Start master
                List<String> masterArgs = List.of(
                    String.valueOf(port), String.valueOf(workers),
                    String.valueOf(steps), String.valueOf(seed));
                Process master = startProcess("com.distributed.mlp.Master", masterArgs, log);
                Thread.sleep(2000);

                // Start workers, track the target worker
                List<Process> workerProcs = new ArrayList<>();
                Process targetWorker = null;
                for (int i = 0; i < workers; i++) {
                    List<String> wArgs = List.of("127.0.0.1", String.valueOf(port),
                        String.valueOf(i), String.valueOf(workers),
                        String.valueOf(steps), String.valueOf(seed));
                    Process wp = startProcess("com.distributed.mlp.Worker", wArgs, log);
                    workerProcs.add(wp);
                    if (i == workerToCrash) targetWorker = wp;
                    Thread.sleep(200);
                }

                // Schedule the kill
                if (targetWorker != null) {
                    final Process toKill = targetWorker;
                    Thread.sleep(crashAfterSec * 1000L);
                    Platform.runLater(() -> log.accept("💥 Killing worker " + workerToCrash + " NOW (simulated crash)!"));
                    toKill.destroyForcibly();
                }

                // Wait for remaining processes
                for (Process wp : workerProcs) {
                    if (wp.isAlive()) wp.waitFor();
                }
                master.waitFor();

                Platform.runLater(() -> {
                    log.accept("✅ Crash test session ended. Master has exited.");
                    onDone.accept(true);
                });
            } catch (Exception ex) {
                Platform.runLater(() -> {
                    log.accept("❌ Crash test error: " + ex.getMessage());
                    onDone.accept(false);
                });
            }
        });
    }
*/

// NOTE: This file is a reference — copy the method above into ProcessManager.java
public class ProcessManagerAddendum {}
