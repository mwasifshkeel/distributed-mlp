package com.distributed.mlp.gui;

import javafx.geometry.*;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;

/**
 * Builds the root layout: left sidebar nav + right content area.
 */
public class DashboardLayout {

    private final Stage stage;
    private final StackPane contentArea = new StackPane();
    private final LogConsole logConsole = new LogConsole();

    // Panels (lazy-init)
    private Node trainingPanel;
    private Node benchmarkPanel;
    private Node scalingPanel;
    private Node crashPanel;
    private Node inferencePanel;
    private Node chartsPanel;

    public DashboardLayout(Stage stage) {
        this.stage = stage;
    }

    public BorderPane build() {
        BorderPane root = new BorderPane();
        root.getStyleClass().add("root-pane");

        // Header bar
        root.setTop(buildHeader());

        // Left nav
        root.setLeft(buildSidebar());

        // Center = content + bottom log
        VBox center = new VBox(0);
        VBox.setVgrow(contentArea, Priority.ALWAYS);
        contentArea.getStyleClass().add("content-area");
        center.getChildren().addAll(contentArea, buildLogPane());
        root.setCenter(center);

        // Default view
        showPanel("training");

        return root;
    }

    private HBox buildHeader() {
        HBox header = new HBox();
        header.getStyleClass().add("header-bar");
        header.setAlignment(Pos.CENTER_LEFT);
        header.setPadding(new Insets(0, 24, 0, 24));

        Label icon = new Label("⬡");
        icon.getStyleClass().add("header-icon");

        Label title = new Label("Distributed MLP");
        title.getStyleClass().add("header-title");

        Label subtitle = new Label("Training Control Center");
        subtitle.getStyleClass().add("header-subtitle");

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        Label status = new Label("● READY");
        status.getStyleClass().addAll("status-badge", "status-ready");
        status.setId("global-status");

        header.getChildren().addAll(icon, title, subtitle, spacer, status);
        return header;
    }

    private VBox buildSidebar() {
        VBox sidebar = new VBox(4);
        sidebar.getStyleClass().add("sidebar");
        sidebar.setPadding(new Insets(16, 8, 16, 8));

        Label sectionLabel = new Label("OPERATIONS");
        sectionLabel.getStyleClass().add("sidebar-section-label");

        sidebar.getChildren().add(sectionLabel);
        sidebar.getChildren().add(navButton("🚀", "Training", "training"));
        sidebar.getChildren().add(navButton("📊", "Benchmarks", "benchmark"));
        sidebar.getChildren().add(navButton("📈", "Scaling", "scaling"));
        sidebar.getChildren().add(navButton("💥", "Crash Test", "crash"));
        sidebar.getChildren().add(navButton("🔍", "Inference", "inference"));

        Label sectionLabel2 = new Label("ANALYSIS");
        sectionLabel2.getStyleClass().add("sidebar-section-label");
        sectionLabel2.setPadding(new Insets(16, 0, 4, 8));
        sidebar.getChildren().add(sectionLabel2);
        sidebar.getChildren().add(navButton("📉", "Charts & Results", "charts"));

        return sidebar;
    }

    private Button navButton(String icon, String label, String panelId) {
        Button btn = new Button(icon + "  " + label);
        btn.getStyleClass().add("nav-btn");
        btn.setMaxWidth(Double.MAX_VALUE);
        btn.setOnAction(e -> showPanel(panelId));
        return btn;
    }

    private void showPanel(String id) {
        Node panel = switch (id) {
            case "training"  -> trainingPanel  == null ? (trainingPanel  = new TrainingPanel(logConsole).build())  : trainingPanel;
            case "benchmark" -> benchmarkPanel == null ? (benchmarkPanel = new BenchmarkPanel(logConsole).build()) : benchmarkPanel;
            case "scaling"   -> scalingPanel   == null ? (scalingPanel   = new ScalingPanel(logConsole).build())   : scalingPanel;
            case "crash"     -> crashPanel     == null ? (crashPanel     = new CrashTestPanel(logConsole).build()) : crashPanel;
            case "inference" -> inferencePanel == null ? (inferencePanel = new InferencePanel(logConsole).build()) : inferencePanel;
            case "charts"    -> chartsPanel    == null ? (chartsPanel    = new ChartsPanel().build())              : chartsPanel;
            default -> new Label("Unknown panel");
        };
        contentArea.getChildren().setAll(panel);
    }

    private Node buildLogPane() {
        TitledPane tp = new TitledPane("Console Output", logConsole.build());
        tp.getStyleClass().add("log-pane");
        tp.setPrefHeight(180);
        tp.setCollapsible(true);
        return tp;
    }
}
