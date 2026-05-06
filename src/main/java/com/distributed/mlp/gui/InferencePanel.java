package com.distributed.mlp.gui;

import javafx.geometry.*;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.image.*;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.*;
import java.nio.file.*;
import java.util.List;

/**
 * Panel: Load a saved model and run inference on CIFAR-10 samples.
 * Displays predicted class, confidence, and a pixel-art rendering of the image.
 */
public class InferencePanel {

    private static final String[] CLASSES = {
        "✈ airplane", "🚗 automobile", "🐦 bird", "🐱 cat", "🦌 deer",
        "🐶 dog", "🐸 frog", "🐴 horse", "🚢 ship", "🚚 truck"
    };

    private final LogConsole log;
    private TextField weightsField;
    private Spinner<Integer> numSamplesSpinner;
    private FlowPane imageGrid;
    private Label accuracyLabel;

    public InferencePanel(LogConsole log) { this.log = log; }

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

        root.getChildren().add(panelHeader("🔍 Inference", "Evaluate a saved model on CIFAR-10 samples"));
        root.getChildren().add(buildConfigSection());
        root.getChildren().add(buildResultsSection());

        return root;
    }

    private Node buildConfigSection() {
        TitledPane tp = new TitledPane("Model & Inference Settings", null);
        tp.setCollapsible(false);
        tp.getStyleClass().add("config-section");

        GridPane grid = new GridPane();
        grid.setHgap(16); grid.setVgap(14);
        grid.setPadding(new Insets(16));

        // Weights file picker
        weightsField = new TextField("results/model_weights_0.bin");
        weightsField.setPrefWidth(320);

        Button browseBtn = new Button("📁 Browse");
        browseBtn.getStyleClass().add("btn-outline");
        browseBtn.setOnAction(e -> {
            FileChooser fc = new FileChooser();
            fc.setTitle("Select Model Weights (.bin)");
            fc.getExtensionFilters().add(new FileChooser.ExtensionFilter("Binary weights", "*.bin"));
            fc.setInitialDirectory(ProcessManager.projectRoot().resolve("results").toFile().exists()
                ? ProcessManager.projectRoot().resolve("results").toFile()
                : ProcessManager.projectRoot().toFile());
            File f = fc.showOpenDialog(null);
            if (f != null) weightsField.setText(f.getAbsolutePath());
        });

        HBox weightsRow = new HBox(8, weightsField, browseBtn);
        weightsRow.setAlignment(Pos.CENTER_LEFT);

        numSamplesSpinner = new Spinner<>(1, 1000, 100);
        numSamplesSpinner.setEditable(true);
        numSamplesSpinner.setPrefWidth(110);

        Label wLabel = new Label("Weights file:");  wLabel.getStyleClass().add("config-label");
        Label nLabel = new Label("# Samples:");     nLabel.getStyleClass().add("config-label");

        grid.add(wLabel, 0, 0); grid.add(weightsRow, 1, 0);
        grid.add(nLabel, 0, 1); grid.add(numSamplesSpinner, 1, 1);

        // Quick-select saved models
        Label savedLabel = new Label("Quick select:");
        savedLabel.getStyleClass().add("config-label");

        HBox savedRow = new HBox(8);
        savedRow.setAlignment(Pos.CENTER_LEFT);
        Button refreshBtn = new Button("🔄 Scan results/");
        refreshBtn.getStyleClass().add("btn-secondary");
        refreshBtn.setOnAction(e -> scanAndPopulateSaved(savedRow));

        savedRow.getChildren().add(refreshBtn);
        scanAndPopulateSaved(savedRow); // initial scan

        grid.add(savedLabel, 0, 2); grid.add(savedRow, 1, 2);

        HBox btnRow = new HBox(12);
        ProgressBar pb = new ProgressBar(0);
        pb.setMaxWidth(Double.MAX_VALUE);
        pb.getStyleClass().add("training-progress");

        Button runBtn = new Button("🔍  Run Inference");
        runBtn.getStyleClass().add("btn-primary");
        runBtn.setOnAction(e -> {
            runBtn.setDisable(true);
            pb.setProgress(-1);
            imageGrid.getChildren().clear();
            accuracyLabel.setText("Running...");
            log.clear();

            String weights = weightsField.getText().trim();
            int n = numSamplesSpinner.getValue();

            // Run Inference class, parse output for image display
            ProcessManager.getInstance().launch(
                "com.distributed.mlp.Inference",
                List.of(weights, String.valueOf(n)),
                line -> {
                    log.asConsumer().accept(line);
                    parsePredictionLine(line);
                },
                code -> {
                    runBtn.setDisable(false);
                    pb.setProgress(code == 0 ? 1.0 : 0);
                }
            );
        });

        btnRow.getChildren().addAll(runBtn);

        VBox box = new VBox(14, grid, btnRow, pb);
        box.setPadding(new Insets(14));
        tp.setContent(box);
        return tp;
    }

    private void scanAndPopulateSaved(HBox row) {
        // Remove old model buttons (keep Refresh button at index 0)
        while (row.getChildren().size() > 1) row.getChildren().remove(1);

        Path results = ProcessManager.projectRoot().resolve("results");
        if (!results.toFile().exists()) return;

        try {
            Files.list(results)
                .filter(p -> p.getFileName().toString().startsWith("model_weights_") && p.toString().endsWith(".bin"))
                .sorted()
                .forEach(p -> {
                    Button b = new Button(p.getFileName().toString());
                    b.getStyleClass().add("btn-outline-small");
                    b.setOnAction(e -> weightsField.setText(p.toString()));
                    row.getChildren().add(b);
                });
        } catch (IOException ignored) {}
    }

    private void parsePredictionLine(String line) {
        // Inference prints lines like:
        // [3] True: cat | Predicted: dog | Confidence: 0.342
        if (line.contains("True:") && line.contains("Predicted:") && line.contains("Confidence:")) {
            try {
                String[] parts = line.split("\\|");
                String trueClass     = parts[0].substring(parts[0].indexOf("True:") + 5).trim();
                String predClass     = parts[1].substring(parts[1].indexOf("Predicted:") + 10).trim();
                double confidence    = Double.parseDouble(parts[2].substring(parts[2].indexOf(':') + 1).trim());
                boolean correct      = trueClass.equalsIgnoreCase(predClass);

                javafx.application.Platform.runLater(() ->
                    imageGrid.getChildren().add(buildPredCard(trueClass, predClass, confidence, correct)));
            } catch (Exception ignored) {}
        }
        // Accuracy line: "Accuracy: 23.40% (234/1000)"
        if (line.startsWith("Accuracy:")) {
            javafx.application.Platform.runLater(() -> accuracyLabel.setText(line));
        }
    }

    private Node buildResultsSection() {
        VBox box = new VBox(12);
        box.getStyleClass().add("results-section");

        HBox header = new HBox(16);
        header.setAlignment(Pos.CENTER_LEFT);
        Label title = new Label("Prediction Results");
        title.getStyleClass().add("section-label");
        accuracyLabel = new Label("No results yet");
        accuracyLabel.getStyleClass().add("accuracy-badge");
        Region spacer = new Region(); HBox.setHgrow(spacer, Priority.ALWAYS);
        header.getChildren().addAll(title, spacer, accuracyLabel);

        imageGrid = new FlowPane(10, 10);
        imageGrid.getStyleClass().add("image-grid");
        imageGrid.setPrefWrapLength(1100);

        Label placeholder = new Label("Run inference to see predictions here.\nEach card shows true class, predicted class, and confidence.");
        placeholder.getStyleClass().add("placeholder-label");
        placeholder.setTextAlignment(javafx.scene.text.TextAlignment.CENTER);
        imageGrid.getChildren().add(placeholder);

        box.getChildren().addAll(header, imageGrid);
        return box;
    }

    private Node buildPredCard(String trueClass, String predClass, double confidence, boolean correct) {
        VBox card = new VBox(6);
        card.getStyleClass().addAll("pred-card", correct ? "pred-correct" : "pred-wrong");
        card.setPrefWidth(130);
        card.setPadding(new Insets(10));
        card.setAlignment(Pos.CENTER);

        // Fake pixel art placeholder (real rendering requires access to raw pixel data from DataLoader)
        javafx.scene.canvas.Canvas canvas = buildPlaceholderCanvas(trueClass, correct);

        Label predLabel = new Label("→ " + mapClassName(predClass));
        predLabel.getStyleClass().add("pred-label");
        predLabel.setWrapText(true);

        Label trueLabel = new Label("✓ " + mapClassName(trueClass));
        trueLabel.getStyleClass().add("true-label");

        Label confLabel = new Label(String.format("%.1f%%", confidence * 100));
        confLabel.getStyleClass().addAll("conf-label", correct ? "conf-high" : "conf-low");

        card.getChildren().addAll(canvas, predLabel, trueLabel, confLabel);
        return card;
    }

    private javafx.scene.canvas.Canvas buildPlaceholderCanvas(String className, boolean correct) {
        javafx.scene.canvas.Canvas c = new javafx.scene.canvas.Canvas(80, 80);
        javafx.scene.canvas.GraphicsContext gc = c.getGraphicsContext2D();

        // Background gradient based on class
        Color bg = classColor(className);
        gc.setFill(bg);
        gc.fillRoundRect(0, 0, 80, 80, 10, 10);

        // Class emoji/icon
        gc.setFill(Color.WHITE);
        gc.setFont(javafx.scene.text.Font.font(32));
        gc.fillText(classEmoji(className), 16, 52);

        // Correct/wrong indicator
        gc.setStroke(correct ? Color.rgb(0, 200, 100) : Color.rgb(255, 80, 80));
        gc.setLineWidth(3);
        gc.strokeRoundRect(1, 1, 78, 78, 10, 10);

        return c;
    }

    private static Color classColor(String cls) {
        return switch (cls.toLowerCase().replaceAll("[^a-z]","")) {
            case "airplane"    -> Color.rgb(30, 120, 210);
            case "automobile"  -> Color.rgb(200, 80, 30);
            case "bird"        -> Color.rgb(100, 180, 60);
            case "cat"         -> Color.rgb(180, 120, 60);
            case "deer"        -> Color.rgb(120, 160, 80);
            case "dog"         -> Color.rgb(160, 100, 40);
            case "frog"        -> Color.rgb(60, 160, 60);
            case "horse"       -> Color.rgb(100, 70, 40);
            case "ship"        -> Color.rgb(30, 80, 160);
            case "truck"       -> Color.rgb(80, 80, 80);
            default            -> Color.rgb(60, 60, 80);
        };
    }

    private static String classEmoji(String cls) {
        return switch (cls.toLowerCase().replaceAll("[^a-z]","")) {
            case "airplane"    -> "✈";
            case "automobile"  -> "🚗";
            case "bird"        -> "🐦";
            case "cat"         -> "🐱";
            case "deer"        -> "🦌";
            case "dog"         -> "🐕";
            case "frog"        -> "🐸";
            case "horse"       -> "🐴";
            case "ship"        -> "⛵";
            case "truck"       -> "🚛";
            default            -> "?";
        };
    }

    private static String mapClassName(String raw) {
        for (String c : CLASSES) {
            if (c.toLowerCase().contains(raw.toLowerCase())) return c;
        }
        return raw;
    }

    private static Node panelHeader(String title, String subtitle) {
        VBox box = new VBox(4);
        Label t = new Label(title); t.getStyleClass().add("panel-title");
        Label s = new Label(subtitle); s.getStyleClass().add("panel-subtitle");
        box.getChildren().addAll(t, s);
        return box;
    }
}