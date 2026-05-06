package com.distributed.mlp.gui;

import java.util.List;

import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TitledPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.VBox;

/**
 * Panel: Run benchmarks (sequential, sync-SGD, full BenchmarkRunner, serializer).
 */
public class BenchmarkPanel {

    private final LogConsole log;
    private ProgressBar progress;

    public BenchmarkPanel(LogConsole log) { this.log = log; }

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

        root.getChildren().add(panelHeader("📊 Benchmarks", "Run individual or full benchmark suites"));

        progress = new ProgressBar(0);
        progress.setMaxWidth(Double.MAX_VALUE);
        progress.getStyleClass().add("training-progress");

        root.getChildren().add(progress);
        root.getChildren().add(buildBenchCards());
        root.getChildren().add(buildFullSuiteSection());

        return root;
    }

    private Node buildBenchCards() {
        Label sectionLabel = new Label("Individual Benchmarks");
        sectionLabel.getStyleClass().add("section-label");

        FlowPane flow = new FlowPane(16, 16);
        flow.setPrefWrapLength(900);

        flow.getChildren().add(benchCard(
            "Sequential Baseline",
            "Single-threaded training on a fixed subset.\nOutputs: results/sequential_results.csv",
            "▶ Run Sequential",
            "com.distributed.mlp.baseline.SequentialBaseline",
            List.of()
        ));

        flow.getChildren().add(benchCard(
            "Sync SGD Baseline",
            "Multi-threaded synchronous SGD with CyclicBarrier.\nOutputs: results/sync_results.csv",
            "▶ Run Sync SGD",
            "com.distributed.mlp.baseline.SyncSGDBaseline",
            List.of()
        ));

        flow.getChildren().add(benchCard(
            "Correctness Checker",
            "Validates sequential-equivalent vs 1-worker distributed output.\nOutputs: results/correctness_model.bin",
            "▶ Check Correctness",
            "com.distributed.mlp.correctness.CorrectnessChecker",
            List.of()
        ));

        flow.getChildren().add(benchCard(
            "Optimization Benchmark",
            "Baseline vs optimized distributed run.\nOutputs: results/optimisation_runs.csv",
            "▶ Run Optimization",
            "com.distributed.mlp.bench.OptimizationBenchmark",
            List.of()
        ));

        flow.getChildren().add(benchCard(
            "Serializer Benchmark",
            "ObjectOutputStream vs ByteBuffer for 402k doubles.\nOutputs: results/serial_bench.csv",
            "▶ Run Serializer Bench",
            "com.distributed.mlp.bench.SerializerBenchmark",
            List.of()
        ));

        flow.getChildren().add(benchCard(
            "Speedup Analyzer",
            "Reads raw.csv, computes speedup/efficiency vs Amdahl.\nOutputs: speedup_table.txt, amdahl_comparison.csv",
            "▶ Analyze Speedup",
            "com.distributed.mlp.bench.SpeedupAnalyzer",
            List.of()
        ));

        flow.getChildren().add(benchCard(
            "Optimisation Report",
            "Compares float32 compression vs double.\nOutputs: results/optimisation_before_after.csv",
            "▶ Generate Report",
            "com.distributed.mlp.optimisation.GradientCompressor",
            List.of()
        ));

        flow.getChildren().add(benchCard(
            "Plot Generator",
            "Generates PNG charts from CSV outputs.\nOutputs: results/plots/*.png",
            "▶ Generate Plots",
            "com.distributed.mlp.bench.BenchPlotter",
            List.of()
        ));

        VBox box = new VBox(10, sectionLabel, flow);
        return box;
    }

    private Node buildFullSuiteSection() {
        TitledPane tp = new TitledPane("Full Benchmark Suite", null);
        tp.setCollapsible(false);
        tp.getStyleClass().add("config-section");

        Label desc = new Label(
            "Runs BenchmarkRunner across all worker configs {1,2,4,8} and input sizes {10k, 40k, 75k}.\n" +
            "Then runs ScalingExperiment (strong + weak) and SpeedupAnalyzer.\n" +
            "⚠ This may take 10–30 minutes depending on hardware.");
        desc.getStyleClass().add("bench-desc");
        desc.setWrapText(true);

        Button runAll = new Button("🏃 Run Full Suite");
        runAll.getStyleClass().add("btn-primary");
        runAll.setOnAction(e -> runFullSuite(runAll));

        VBox box = new VBox(12, desc, runAll);
        box.setPadding(new Insets(14));
        tp.setContent(box);
        return tp;
    }

    private Node benchCard(String title, String desc, String btnLabel, String mainClass, List<String> args) {
        VBox card = new VBox(10);
        card.getStyleClass().add("bench-card");
        card.setPrefWidth(260);

        Label t = new Label(title);
        t.getStyleClass().add("card-title");

        Label d = new Label(desc);
        d.getStyleClass().add("card-desc");
        d.setWrapText(true);

        Button btn = new Button(btnLabel);
        btn.getStyleClass().add("btn-outline");
        btn.setMaxWidth(Double.MAX_VALUE);
        btn.setOnAction(e -> {
            btn.setDisable(true);
            progress.setProgress(-1);
            log.append("Starting " + title + "...");
            ProcessManager.getInstance().launch(mainClass, args, log.asConsumer(), code -> {
                btn.setDisable(false);
                progress.setProgress(code == 0 ? 1.0 : 0);
                log.append(code == 0 ? "✅ " + title + " complete." : "❌ " + title + " failed (exit " + code + ")");
            });
        });

        card.getChildren().addAll(t, d, btn);
        return card;
    }

    private void runFullSuite(Button btn) {
        btn.setDisable(true);
        progress.setProgress(-1);
        log.clear();
        log.append("▶ Starting full benchmark suite...");

        ProcessManager.getInstance().launch(
            "com.distributed.mlp.bench.BenchmarkRunner",
            List.of(),
            log.asConsumer(),
            code1 -> {
                if (code1 != 0) { btn.setDisable(false); progress.setProgress(0); return; }
                log.append("BenchmarkRunner done. Starting ScalingExperiment...");
                ProcessManager.getInstance().launch(
                    "com.distributed.mlp.bench.ScalingExperiment",
                    List.of(),
                    log.asConsumer(),
                    code2 -> {
                        if (code2 != 0) { btn.setDisable(false); progress.setProgress(0); return; }
                        log.append("ScalingExperiment done. Analyzing speedup...");
                        ProcessManager.getInstance().launch(
                            "com.distributed.mlp.bench.SpeedupAnalyzer",
                            List.of(),
                            log.asConsumer(),
                            code3 -> {
                                btn.setDisable(false);
                                progress.setProgress(code3 == 0 ? 1.0 : 0);
                                log.append(code3 == 0 ? "✅ Full suite complete! Check Charts tab." : "❌ SpeedupAnalyzer failed.");
                            });
                    });
            });
    }

    private static Node panelHeader(String title, String subtitle) {
        VBox box = new VBox(4);
        Label t = new Label(title); t.getStyleClass().add("panel-title");
        Label s = new Label(subtitle); s.getStyleClass().add("panel-subtitle");
        box.getChildren().addAll(t, s);
        return box;
    }
}
