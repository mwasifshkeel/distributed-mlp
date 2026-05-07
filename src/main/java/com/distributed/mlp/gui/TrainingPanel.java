package com.distributed.mlp.gui;

import com.distributed.mlp.model.MLPModel;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Spinner;
import javafx.scene.control.TextField;
import javafx.scene.control.TitledPane;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;

/**
 * Panel: Configure and launch distributed training.
 */
public class TrainingPanel {

    private final LogConsole log;
    private Label statusDot;
    private Label statusLabel;
    private ProgressBar progressBar;
    private Button startBtn, stopBtn;

    // Constants matching CIFAR-10 and run.sh
    private static final int TOTAL_SAMPLES = 50_000;
    private static final int MINI_BATCH_SIZE = 32;

    private Spinner<Integer> workersSpinner;
    private Spinner<Integer> epochsSpinner;
    private TextField stepsAutoField;
    private Spinner<Integer> computeThreadsSpinner;
    private Spinner<Integer> sampleLimitSpinner;
    private ChoiceBox<String> modeChoice;

    public TrainingPanel(LogConsole log) {
        this.log = log;
    }

    public Node build() {
        ScrollPane scroll = new ScrollPane(buildContent());
        scroll.setFitToWidth(true);
        scroll.getStyleClass().add("panel-scroll");
        return scroll;
    }

    private Node buildContent() {
        VBox root = new VBox(20);
        root.getStyleClass().add("panel-content");
        root.setPadding(new Insets(28, 32, 28, 32));

        root.getChildren().add(panelHeader("🚀 Distributed Training", "Configure and launch Master + Worker nodes"));
        root.getChildren().add(buildConfigSection());
        root.getChildren().add(buildControlSection());
        root.getChildren().add(buildStatusSection());

        return root;
    }

    // ── Config ────────────────────────────────────────────────────────────────

    private Spinner<Integer> portSpinner, seedSpinner;

    private Node buildConfigSection() {
        TitledPane tp = new TitledPane();
        tp.setText("Configuration");
        tp.getStyleClass().add("config-section");
        tp.setCollapsible(false);

        GridPane grid = new GridPane();
        grid.setHgap(24);
        grid.setVgap(14);
        grid.setPadding(new Insets(16));

        workersSpinner = intSpinner(1, 16, 3);
        epochsSpinner  = intSpinner(1, 20, 5);
        portSpinner    = intSpinner(1024, 65535, 9000);
        seedSpinner    = intSpinner(0, 99999, 42);
        computeThreadsSpinner = intSpinner(1, 64,
            Math.max(1, Runtime.getRuntime().availableProcessors() - 1));

        modeChoice = new ChoiceBox<>();
        modeChoice.getItems().addAll(
            "Standard (run.sh)",
            "Baseline (run_baseline.sh)",
            "Optimised (run_optimised.sh)"
        );
        modeChoice.getSelectionModel().selectFirst();

        addRow(grid, 0, "Workers",        workersSpinner, "Number of Worker JVMs to launch");
        addRow(grid, 1, "Epochs",         epochsSpinner,  "Number of passes over the dataset");
        sampleLimitSpinner = intSpinner(0, TOTAL_SAMPLES, 0);
        addRow(grid, 2, "Sample Limit", sampleLimitSpinner, "0 = full dataset; otherwise use N samples total");
        // Auto‑calculated steps per worker (read-only)
        stepsAutoField = new TextField();
        stepsAutoField.setEditable(false);
        stepsAutoField.getStyleClass().add("formula-field");  // defined in CSS
        GridPane.setHgrow(stepsAutoField, Priority.ALWAYS);
        updateAutoSteps();  // initial value
        addRow(grid, 3, "→ Steps/Worker", stepsAutoField, "Auto: (samples / workers / 32) * epochs");
        addRow(grid, 4, "Master Port",    portSpinner,     "TCP port for the parameter server");
        addRow(grid, 5, "Random Seed",    seedSpinner,     "Seed for deterministic sharding & weight init");
        addRow(grid, 6, "Compute Threads", computeThreadsSpinner, "Per-worker compute threads");
        addRow(grid, 7, "Mode",           modeChoice, "Matches provided shell scripts");

        // Listeners to update steps whenever workers or epochs change
        workersSpinner.valueProperty().addListener((obs, old, val) -> updateAutoSteps());
        epochsSpinner.valueProperty().addListener((obs, old, val) -> updateAutoSteps());
        sampleLimitSpinner.valueProperty().addListener((obs, old, val) -> updateAutoSteps());

        // Architecture info (read-only)
        Label archLabel = new Label("Architecture:  3072 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)   |   Xavier init   |   Async SGD   |   Mini-batch 32");
        archLabel.getStyleClass().add("arch-info");

        VBox box = new VBox(12, grid, archLabel);
        box.setPadding(new Insets(0, 0, 4, 0));
        tp.setContent(box);
        return tp;
    }

    private void updateAutoSteps() {
        int workers = workersSpinner.getValue();
        int epochs  = epochsSpinner.getValue();
        int totalSamples = getEffectiveTotalSamples();
        int stepsPerWorker = Math.max(1, (totalSamples / workers / MINI_BATCH_SIZE) * epochs);
        stepsAutoField.setText(String.valueOf(stepsPerWorker));
    }

    private int getEffectiveTotalSamples() {
        int limit = sampleLimitSpinner.getValue();
        if (limit <= 0) {
            return TOTAL_SAMPLES;
        }
        return Math.min(limit, TOTAL_SAMPLES);
    }

    private Node buildControlSection() {
        HBox box = new HBox(12);
        box.setAlignment(Pos.CENTER_LEFT);

        startBtn = new Button("▶  Start Training");
        startBtn.getStyleClass().addAll("btn-primary");
        startBtn.setOnAction(e -> startTraining());

        stopBtn = new Button("■  Stop All");
        stopBtn.getStyleClass().add("btn-danger");
        stopBtn.setDisable(true);
        stopBtn.setOnAction(e -> stopTraining());

        Button clearBtn = new Button("🗑  Clear Log");
        clearBtn.getStyleClass().add("btn-secondary");
        clearBtn.setOnAction(e -> log.clear());

        box.getChildren().addAll(startBtn, stopBtn, clearBtn);
        return box;
    }

    private Node buildStatusSection() {
        VBox box = new VBox(10);
        box.getStyleClass().add("status-section");

        HBox statusRow = new HBox(10);
        statusRow.setAlignment(Pos.CENTER_LEFT);
        statusDot = new Label("●");
        statusDot.getStyleClass().addAll("status-dot", "dot-idle");
        statusLabel = new Label("Idle");
        statusLabel.getStyleClass().add("status-text");
        statusRow.getChildren().addAll(statusDot, statusLabel);

        progressBar = new ProgressBar(0);
        progressBar.getStyleClass().add("training-progress");
        progressBar.setMaxWidth(Double.MAX_VALUE);
        progressBar.setPrefHeight(8);

        // Info cards
        HBox cards = new HBox(12);
        cards.getChildren().addAll(
            infoCard("Target Class", "10 (CIFAR-10)"),
            infoCard("Input Dim", "3072 (32×32×3)"),
            infoCard("Parameters", String.format("%,d", MLPModel.parameterCount())),
            infoCard("Protocol", "Async TCP Push/Pull"),
            infoCard("Compression", "float32 gradients")
        );

        box.getChildren().addAll(statusRow, progressBar, cards);
        return box;
    }

    // ── Actions ───────────────────────────────────────────────────────────────

    private void startTraining() {
        int workers = workersSpinner.getValue();
        // Use the auto‑computed steps – already guaranteed correct
        int steps   = Integer.parseInt(stepsAutoField.getText());
        int port    = portSpinner.getValue();
        int seed    = seedSpinner.getValue();
        int totalSamples = getEffectiveTotalSamples();

        int computeThreads = computeThreadsSpinner.getValue();
        String mode = modeChoice.getSelectionModel().getSelectedItem();

        setStatus("running", "Training... (" + workers + " workers, " + steps + " steps each)");
        progressBar.setProgress(-1); // indeterminate
        startBtn.setDisable(true);
        stopBtn.setDisable(false);
        log.clear();
        log.append("Starting distributed training: workers=" + workers + ", steps=" + steps
            + ", port=" + port + ", seed=" + seed + ", computeThreads=" + computeThreads
            + ", totalSamples=" + totalSamples);

        String script = switch (mode) {
            case "Baseline (run_baseline.sh)" -> "run_baseline.sh";
            case "Optimised (run_optimised.sh)" -> "run_optimised.sh";
            default -> "run.sh";
        };

        java.util.Map<String, String> env = new java.util.HashMap<>();
        env.put("MLP_COMPUTE_THREADS", String.valueOf(computeThreads));
        if (totalSamples < TOTAL_SAMPLES) {
            env.put("MLP_MAX_SAMPLES", String.valueOf(totalSamples));
            env.put("MLP_TOTAL_SAMPLES", String.valueOf(totalSamples));
        }

        ProcessManager.getInstance().launchScript(
            script,
            java.util.List.of(String.valueOf(port), String.valueOf(workers), String.valueOf(computeThreads)),
            env,
            log.asConsumer(),
            code -> {
                boolean success = code == 0;
                setStatus(success ? "done" : "error", success ? "Training Complete ✓" : "Training Failed ✗");
                progressBar.setProgress(success ? 1.0 : 0);
                startBtn.setDisable(false);
                stopBtn.setDisable(true);
                deleteStaleCheckpoints();
            }
        );
    }

    private void stopTraining() {
        ProcessManager.getInstance().killAll();
        setStatus("idle", "Stopped");
        progressBar.setProgress(0);
        startBtn.setDisable(false);
        stopBtn.setDisable(true);
        log.append("⏹ All processes killed.");
    }

    private void setStatus(String type, String text) {
        statusDot.getStyleClass().removeAll("dot-idle", "dot-running", "dot-done", "dot-error");
        statusDot.getStyleClass().add("dot-" + type);
        statusLabel.setText(text);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void addRow(GridPane g, int row, String label, Node ctrl, String tooltip) {
        Label lbl = new Label(label);
        lbl.getStyleClass().add("config-label");
        Tooltip.install(ctrl, new Tooltip(tooltip));
        g.add(lbl, 0, row);
        g.add(ctrl, 1, row);
    }

    private static Spinner<Integer> intSpinner(int min, int max, int def) {
        Spinner<Integer> s = new Spinner<>(min, max, def);
        s.setEditable(true);
        s.setPrefWidth(110);
        return s;
    }

    private static Node panelHeader(String title, String subtitle) {
        VBox box = new VBox(4);
        Label t = new Label(title);
        t.getStyleClass().add("panel-title");
        Label s = new Label(subtitle);
        s.getStyleClass().add("panel-subtitle");
        box.getChildren().addAll(t, s);
        return box;
    }

    private static Node infoCard(String key, String value) {
        VBox card = new VBox(4);
        card.getStyleClass().add("info-card");
        card.setPadding(new Insets(10, 16, 10, 16));
        Label k = new Label(key);
        k.getStyleClass().add("card-key");
        Label v = new Label(value);
        v.getStyleClass().add("card-value");
        card.getChildren().addAll(k, v);
        return card;
    }

    /** Delete all checkpoint_*.bin files from the results/ folder. */
    private void deleteStaleCheckpoints() {
        try {
            java.nio.file.Path dir = java.nio.file.Path.of("results");
            if (!java.nio.file.Files.exists(dir)) return;
            try (java.nio.file.DirectoryStream<java.nio.file.Path> stream =
                java.nio.file.Files.newDirectoryStream(dir, "checkpoint_*.bin")) {
                for (java.nio.file.Path p : stream) {
                    java.nio.file.Files.deleteIfExists(p);
                }
            }
            log.append("✓ Stale checkpoints removed.");
        } catch (Exception e) {
            log.append("⚠ Could not remove checkpoints: " + e.getMessage());
        }
    }
}