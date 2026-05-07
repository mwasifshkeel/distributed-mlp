package com.distributed.mlp.gui;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.VBox;

/**
 * Panel: Read result CSVs and render interactive charts.
 * Charts: Strong scaling speedup, weak scaling efficiency,
 *         sequential vs distributed loss, serializer comparison,
 *         optimisation before/after, Amdahl's law.
 */
public class ChartsPanel {

    public Node build() {
        ScrollPane scroll = new ScrollPane(buildContent());
        scroll.setFitToWidth(true);
        scroll.getStyleClass().add("panel-scroll");
        return scroll;
    }

    private Node buildContent() {
        VBox root = new VBox(24);
        root.getStyleClass().add("panel-content");
        root.setPadding(new Insets(28, 32, 28, 32));

        root.getChildren().add(panelHeader("📉 Charts & Results",
            "Live charts generated from results/ CSV files — click Refresh to reload after running experiments"));

        Button refreshAll = new Button("🔄 Refresh All Charts");
        refreshAll.getStyleClass().add("btn-primary");
        refreshAll.setOnAction(e -> {
            root.getChildren().removeIf(n -> n.getStyleClass().contains("chart-section"));
            addCharts(root);
        });
        root.getChildren().add(refreshAll);

        addCharts(root);
        return root;
    }

    private void addCharts(VBox root) {
        root.getChildren().add(chartSection("Strong Scaling — Speedup",
            buildStrongScalingChart()));
        root.getChildren().add(chartSection("Weak Scaling — Efficiency",
            buildWeakScalingChart()));
        root.getChildren().add(chartSection("Sequential Training — Loss per Epoch",
            buildSequentialLossChart()));
        root.getChildren().add(chartSection("Sync SGD — Loss per Epoch",
            buildSyncLossChart()));
        root.getChildren().add(chartSection("Amdahl's Law — Measured vs Theoretical",
            buildAmdahlChart()));
        root.getChildren().add(chartSection("Serializer Benchmark — Throughput",
            buildSerializerChart()));
        root.getChildren().add(chartSection("Optimisation — Payload Size Before/After",
            buildOptimisationChart()));
        root.getChildren().add(chartSection("Saved Plots (results/plots)",
            buildSavedPlotsGallery()));
    }

    // ── Chart builders ───────────────────────────────────────────────────────

    /** results/strong_scaling.csv: experiment,workers,input_size,epoch,wall_sec,speedup,efficiency,parallel_fraction */
    private Node buildStrongScalingChart() {
        List<String[]> rows = loadCsv("strong_scaling.csv");
        if (rows.isEmpty()) return placeholder("Run ScalingExperiment first to generate strong_scaling.csv");

        // Speedup line: aggregate by workers (average across epochs)
        Map<Integer, List<Double>> speedupByWorkers = new LinkedHashMap<>();
        for (String[] r : rows) {
            try {
                int w = Integer.parseInt(r[1]);
                if (w <= 0 || w > 8) {
                    continue;
                }
                double sp = Double.parseDouble(r[5]);
                speedupByWorkers.computeIfAbsent(w, k -> new ArrayList<>()).add(sp);
            } catch (Exception ignored) {}
        }

        NumberAxis xAxis = new NumberAxis(1, 8, 1);
        xAxis.setAutoRanging(false);
        xAxis.setLabel("Workers");
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Speedup");

        LineChart<Number, Number> chart = new LineChart<>(xAxis, yAxis);
        chart.setTitle("Strong Scaling Speedup");
        chart.setCreateSymbols(true);
        chart.setPrefHeight(350);

        // Measured
        XYChart.Series<Number, Number> measured = new XYChart.Series<>();
        measured.setName("Measured Speedup");
        // Ideal
        XYChart.Series<Number, Number> ideal = new XYChart.Series<>();
        ideal.setName("Ideal (linear)");

        for (Map.Entry<Integer, List<Double>> e : speedupByWorkers.entrySet()) {
            int w = e.getKey();
            double avgSp = e.getValue().stream().mapToDouble(d->d).average().orElse(0);
            measured.getData().add(new XYChart.Data<>(w, avgSp));
            ideal.getData().add(new XYChart.Data<>(w, w));
        }
        chart.getData().addAll(measured, ideal);
        chart.getStyleClass().add("themed-chart");
        return chart;
    }

    /** results/weak_scaling.csv: experiment,workers,input_size,epoch,wall_sec,speedup,efficiency,parallel_fraction */
    private Node buildWeakScalingChart() {
        List<String[]> rows = loadCsv("weak_scaling.csv");
        if (rows.isEmpty()) return placeholder("Run ScalingExperiment first to generate weak_scaling.csv");

        Map<Integer, List<Double>> effByWorkers = new LinkedHashMap<>();
        for (String[] r : rows) {
            try {
                int w = Integer.parseInt(r[1]);
                if (w <= 0 || w > 8) {
                    continue;
                }
                double eff = Double.parseDouble(r[6]);
                effByWorkers.computeIfAbsent(w, k -> new ArrayList<>()).add(eff);
            } catch (Exception ignored) {}
        }

        CategoryAxis xAxis = new CategoryAxis();
        xAxis.setLabel("Workers");
        xAxis.setCategories(javafx.collections.FXCollections.observableArrayList(
            List.of("1", "2", "3", "4", "5", "6", "7", "8")));
        NumberAxis yAxis = new NumberAxis(0, 1.2, 0.1);
        yAxis.setLabel("Efficiency");

        BarChart<String, Number> chart = new BarChart<>(xAxis, yAxis);
        chart.setTitle("Weak Scaling Efficiency");
        chart.setPrefHeight(350);

        XYChart.Series<String, Number> series = new XYChart.Series<>();
        series.setName("Efficiency");
        for (Map.Entry<Integer, List<Double>> e : effByWorkers.entrySet()) {
            double avg = e.getValue().stream().mapToDouble(d->d).average().orElse(0);
            series.getData().add(new XYChart.Data<>(String.valueOf(e.getKey()), avg));
        }
        chart.getData().add(series);
        chart.getStyleClass().add("themed-chart");
        return chart;
    }

    /** results/sequential_results.csv: epoch,loss,accuracy,wall_sec */
    private Node buildSequentialLossChart() {
        return buildEpochLossChart("sequential_results.csv", "Sequential Baseline Loss");
    }

    private Node buildSyncLossChart() {
        return buildEpochLossChart("sync_results.csv", "Sync SGD Baseline Loss");
    }

    private Node buildEpochLossChart(String filename, String title) {
        List<String[]> rows = loadCsv(filename);
        if (rows.isEmpty()) return placeholder("Run the baseline first to generate " + filename);

        NumberAxis xAxis = new NumberAxis();
        xAxis.setLabel("Epoch");
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Loss");

        LineChart<Number, Number> chart = new LineChart<>(xAxis, yAxis);
        chart.setTitle(title);
        chart.setCreateSymbols(true);
        chart.setPrefHeight(320);

        XYChart.Series<Number, Number> lossSeries = new XYChart.Series<>();
        lossSeries.setName("Cross-Entropy Loss");
        XYChart.Series<Number, Number> accSeries = new XYChart.Series<>();
        accSeries.setName("Accuracy");

        for (String[] r : rows) {
            try {
                int epoch   = Integer.parseInt(r[0]);
                double loss = Double.parseDouble(r[1]);
                double acc  = Double.parseDouble(r[2]);
                lossSeries.getData().add(new XYChart.Data<>(epoch, loss));
                accSeries.getData().add(new XYChart.Data<>(epoch, acc));
            } catch (Exception ignored) {}
        }
        chart.getData().addAll(lossSeries, accSeries);
        chart.getStyleClass().add("themed-chart");
        return chart;
    }

    /** results/amdahl_comparison.csv: workers,measured_speedup,amdahl_estimated,amdahl_f_0_5,amdahl_f_0_9,amdahl_f_0_99,karp_flatt_serial_fraction,gustafson_speedup */
    private Node buildAmdahlChart() {
        List<String[]> rows = loadCsv("amdahl_comparison.csv");
        if (rows.isEmpty()) return placeholder("Run SpeedupAnalyzer to generate amdahl_comparison.csv");

        NumberAxis xAxis = new NumberAxis(1, 8, 1);
        xAxis.setAutoRanging(false);
        xAxis.setLabel("Workers");
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Speedup");

        LineChart<Number, Number> chart = new LineChart<>(xAxis, yAxis);
        chart.setTitle("Amdahl's Law Analysis");
        chart.setPrefHeight(350);

        XYChart.Series<Number, Number> measured = new XYChart.Series<>();
        measured.setName("Measured Speedup");
        XYChart.Series<Number, Number> f50 = new XYChart.Series<>();
        f50.setName("Expected (Amdahl f=0.5)");
        XYChart.Series<Number, Number> gustafson = new XYChart.Series<>();
        gustafson.setName("Gustafson");

        for (String[] r : rows) {
            try {
                int w = Integer.parseInt(r[0]);
                if (w <= 0 || w > 8) {
                    continue;
                }
                    double measuredSpeedup = Double.parseDouble(r[1]);
                    double conservativeSpeedup = Double.parseDouble(r[3]);
                    measured.getData().add(new XYChart.Data<>(w, measuredSpeedup));
                    f50.getData().add(new XYChart.Data<>(w, conservativeSpeedup));
                if (r.length > 7) {
                    gustafson.getData().add(new XYChart.Data<>(w, Double.parseDouble(r[7])));
                }
            } catch (Exception ignored) {}
        }
        chart.getData().addAll(measured, f50, gustafson);
        chart.getStyleClass().add("themed-chart");
        return chart;
    }

    /** results/serial_bench.csv: method,round,time_ms */
    private Node buildSerializerChart() {
        List<String[]> rows = loadCsv("serial_bench.csv");
        if (rows.isEmpty()) return placeholder("Run SerializerBenchmark to generate serial_bench.csv");

        Map<String, List<Double>> timesByMethod = new LinkedHashMap<>();
        for (String[] r : rows) {
            try {
                timesByMethod.computeIfAbsent(r[0], k -> new ArrayList<>()).add(Double.parseDouble(r[2]));
            } catch (Exception ignored) {}
        }

        CategoryAxis xAxis = new CategoryAxis();
        xAxis.setLabel("Serializer Method");
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Avg Time (ms)");

        BarChart<String, Number> chart = new BarChart<>(xAxis, yAxis);
        chart.setTitle("Serializer Performance");
        chart.setPrefHeight(300);

        XYChart.Series<String, Number> series = new XYChart.Series<>();
        series.setName("Avg Time (ms)");
        for (Map.Entry<String, List<Double>> e : timesByMethod.entrySet()) {
            double avg = e.getValue().stream().mapToDouble(d->d).average().orElse(0);
            series.getData().add(new XYChart.Data<>(e.getKey(), avg));
        }
        chart.getData().add(series);
        chart.getStyleClass().add("themed-chart");
        return chart;
    }

    /**
     * results/optimisation_before_after.csv: metric,before,after,improvement_pct
     * Metrics include push_payload_mb, pull_frequency, comm_overhead_pct, wall_sec, accuracy
     */
    private Node buildOptimisationChart() {
        List<String[]> rows = loadCsv("optimisation_before_after.csv");
        if (rows.isEmpty()) return placeholder("Run GradientCompressor to generate optimisation_before_after.csv");

        CategoryAxis xAxis = new CategoryAxis();
        xAxis.setLabel("Metric");
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Value");

        BarChart<String, Number> chart = new BarChart<>(xAxis, yAxis);
        chart.setTitle("Optimisation: Before vs After");
        chart.setPrefHeight(350);

        XYChart.Series<String, Number> before = new XYChart.Series<>(); before.setName("Before");
        XYChart.Series<String, Number> after  = new XYChart.Series<>(); after.setName("After (float32 + lazy pull)");

        for (String[] r : rows) {
            try {
                before.getData().add(new XYChart.Data<>(r[0], Double.parseDouble(r[1])));
                after.getData().add(new XYChart.Data<>(r[0], Double.parseDouble(r[2])));
            } catch (Exception ignored) {}
        }
        chart.getData().addAll(before, after);
        chart.getStyleClass().add("themed-chart");
        return chart;
    }

    private Node buildSavedPlotsGallery() {
        Path plotsDir = ProcessManager.projectRoot().resolve("results").resolve("plots");
        if (!Files.isDirectory(plotsDir)) {
            return placeholder("No results/plots directory found.");
        }

        FlowPane flow = new FlowPane(16, 16);
        flow.setPrefWrapLength(1000);

        try {
            List<Path> images = Files.list(plotsDir)
                .filter(p -> {
                    String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
                    return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".gif");
                })
                .sorted()
                .toList();

            if (images.isEmpty()) {
                return placeholder("No plot images found in results/plots.");
            }

            for (Path imgPath : images) {
                VBox card = new VBox(8);
                card.getStyleClass().add("plot-card");
                card.setPadding(new Insets(10));

                Image image = new Image(imgPath.toUri().toString(), 420, 0, true, true);
                ImageView view = new ImageView(image);
                view.setPreserveRatio(true);
                view.setSmooth(true);

                Label label = new Label(imgPath.getFileName().toString());
                label.getStyleClass().add("card-desc");

                card.getChildren().addAll(view, label);
                flow.getChildren().add(card);
            }
        } catch (IOException e) {
            return placeholder("Error reading plots: " + e.getMessage());
        }

        return flow;
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    private List<String[]> loadCsv(String filename) {
        Path path = ProcessManager.projectRoot().resolve("results").resolve(filename);
        List<String[]> rows = new ArrayList<>();
        if (!path.toFile().exists()) return rows;
        try (BufferedReader br = new BufferedReader(new FileReader(path.toFile()))) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null) {
                if (first) { first = false; continue; } // skip header
                String[] parts = line.split(",");
                if (parts.length > 0) rows.add(parts);
            }
        } catch (IOException ignored) {}
        return rows;
    }

    private static Node chartSection(String title, Node chart) {
        VBox box = new VBox(8);
        box.getStyleClass().add("chart-section");

        Label lbl = new Label(title);
        lbl.getStyleClass().add("section-label");

        box.getChildren().addAll(lbl, chart);
        return box;
    }

    private static Node placeholder(String msg) {
        Label l = new Label("ℹ  " + msg);
        l.getStyleClass().add("placeholder-label");
        l.setPadding(new Insets(30));
        l.setWrapText(true);
        VBox box = new VBox(l);
        box.getStyleClass().add("placeholder-box");
        box.setAlignment(Pos.CENTER);
        box.setPrefHeight(200);
        return box;
    }

    private static Node panelHeader(String title, String subtitle) {
        VBox box = new VBox(4);
        Label t = new Label(title); t.getStyleClass().add("panel-title");
        Label s = new Label(subtitle); s.getStyleClass().add("panel-subtitle");
        box.getChildren().addAll(t, s);
        return box;
    }
}
