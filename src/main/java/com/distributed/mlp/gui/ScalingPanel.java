package com.distributed.mlp.gui;

import java.util.List;

import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Spinner;
import javafx.scene.control.TitledPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;

/**
 * Panel: Strong & weak scaling experiments.
 */
public class ScalingPanel {

    private final LogConsole log;

    public ScalingPanel(LogConsole log) { this.log = log; }

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

        root.getChildren().add(panelHeader("📈 Scaling Experiments",
            "Strong scaling (fixed problem, more workers) and weak scaling (fixed work per worker)"));

        root.getChildren().add(buildExplainerCards());
        root.getChildren().add(buildRunSection());

        return root;
    }

    private Node buildExplainerCards() {
        HBox row = new HBox(16);

        VBox strong = explainerCard(
            "Strong Scaling",
            "Fixed total dataset size\nWorkers: {1, 2, 3, 4, 8}\nMetrics: Speedup, Efficiency, Amdahl fraction\nOutput: strong_scaling.csv",
            "📊 Ideal: Linear speedup (efficiency ≈ 1.0)"
        );

        VBox weak = explainerCard(
            "Weak Scaling",
            "Fixed work per worker\nProblem grows with worker count\nMetrics: Wall time ratio vs 1-worker baseline\nOutput: weak_scaling.csv",
            "📊 Ideal: Constant wall time (efficiency ≈ 1.0)"
        );

        VBox amdahl = explainerCard(
            "Amdahl's Law Analysis",
            "Compares measured speedup to a conservative Amdahl expectation\nExpected line uses f=0.5 and stays near 1-2x\nOutput: amdahl_comparison.csv, speedup_table.txt",
            "📊 Shows serial bottleneck quantification"
        );

        row.getChildren().addAll(strong, weak, amdahl);
        HBox.setHgrow(strong, Priority.ALWAYS);
        HBox.setHgrow(weak, Priority.ALWAYS);
        HBox.setHgrow(amdahl, Priority.ALWAYS);
        return row;
    }

    private Node buildRunSection() {
        TitledPane tp = new TitledPane("Run Scaling Experiments", null);
        tp.setCollapsible(false);
        tp.getStyleClass().add("config-section");

        GridPane grid = new GridPane();
        grid.setHgap(24); grid.setVgap(14);
        grid.setPadding(new Insets(16));

        Label epLabel = new Label("Epochs per config:");
        epLabel.getStyleClass().add("config-label");
        Spinner<Integer> epochsSpinner = new Spinner<>(1, 20, 5);
        epochsSpinner.setEditable(true); epochsSpinner.setPrefWidth(110);

        Label seedLabel = new Label("Seed:");
        seedLabel.getStyleClass().add("config-label");
        Spinner<Integer> seedSpinner = new Spinner<>(0, 99999, 42);
        seedSpinner.setEditable(true); seedSpinner.setPrefWidth(110);

        grid.add(epLabel, 0, 0); grid.add(epochsSpinner, 1, 0);
        grid.add(seedLabel, 0, 1); grid.add(seedSpinner, 1, 1);

        ProgressBar pb = new ProgressBar(0);
        pb.setMaxWidth(Double.MAX_VALUE);
        pb.getStyleClass().add("training-progress");

        HBox btnRow = new HBox(12);

        Button strongBtn = new Button("▶ Strong Scaling");
        strongBtn.getStyleClass().add("btn-outline");
        strongBtn.setOnAction(e -> runExperiment("com.distributed.mlp.bench.ScalingExperiment",
            List.of("strong"), strongBtn, pb, "Strong scaling"));

        Button weakBtn = new Button("▶ Weak Scaling");
        weakBtn.getStyleClass().add("btn-outline");
        weakBtn.setOnAction(e -> runExperiment("com.distributed.mlp.bench.ScalingExperiment",
            List.of("weak"), weakBtn, pb, "Weak scaling"));

        Button bothBtn = new Button("▶ Run Both");
        bothBtn.getStyleClass().add("btn-primary");
        bothBtn.setOnAction(e -> runExperiment("com.distributed.mlp.bench.ScalingExperiment",
            List.of(), bothBtn, pb, "Full scaling experiment"));

        Button analyzeBtn = new Button("📈 Analyze Speedup");
        analyzeBtn.getStyleClass().add("btn-secondary");
        analyzeBtn.setOnAction(e -> runExperiment("com.distributed.mlp.bench.SpeedupAnalyzer",
            List.of(), analyzeBtn, pb, "Speedup analysis"));

        btnRow.getChildren().addAll(strongBtn, weakBtn, bothBtn, analyzeBtn);

        VBox box = new VBox(14, grid, btnRow, pb);
        box.setPadding(new Insets(14));
        tp.setContent(box);
        return tp;
    }

    private void runExperiment(String cls, List<String> args, Button btn, ProgressBar pb, String name) {
        btn.setDisable(true);
        pb.setProgress(-1);
        log.append("Starting " + name + "...");
        ProcessManager.getInstance().launch(cls, args, log.asConsumer(), code -> {
            btn.setDisable(false);
            pb.setProgress(code == 0 ? 1.0 : 0);
            log.append(code == 0 ? "✅ " + name + " complete. Check Charts tab." : "❌ " + name + " failed.");
        });
    }

    private static VBox explainerCard(String title, String body, String note) {
        VBox card = new VBox(8);
        card.getStyleClass().add("explainer-card");
        card.setPadding(new Insets(16));

        Label t = new Label(title); t.getStyleClass().add("card-title");
        Label b = new Label(body);  b.getStyleClass().add("card-desc"); b.setWrapText(true);
        Label n = new Label(note);  n.getStyleClass().add("card-note");  n.setWrapText(true);

        card.getChildren().addAll(t, b, n);
        return card;
    }

    private static Node panelHeader(String title, String subtitle) {
        VBox box = new VBox(4);
        Label t = new Label(title); t.getStyleClass().add("panel-title");
        Label s = new Label(subtitle); s.getStyleClass().add("panel-subtitle");
        box.getChildren().addAll(t, s);
        return box;
    }
}
