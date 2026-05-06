package com.distributed.mlp.gui;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 * Entry point for the Distributed MLP Training Dashboard.
 */
public class MLPDashboard extends Application {

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Distributed MLP Control Center");
        primaryStage.setMinWidth(1300);
        primaryStage.setMinHeight(800);

        DashboardLayout layout = new DashboardLayout(primaryStage);
        Scene scene = new Scene(layout.build(), 1400, 900);
        scene.getStylesheets().add(getClass().getResource("/styles/dashboard.css").toExternalForm());

        primaryStage.setScene(scene);
        primaryStage.show();

        primaryStage.setOnCloseRequest(e -> {
            ProcessManager.getInstance().killAll();
            cleanLogsAndCheckpoints();
            Platform.exit();
        });
    }

    /** Override stop() for graceful cleanup when closing via code as well. */
    @Override
    public void stop() throws Exception {
        ProcessManager.getInstance().killAll();
        cleanLogsAndCheckpoints();
        super.stop();
    }

    private void cleanLogsAndCheckpoints() {
        // Delete any leftover checkpoints from results/
        try {
            java.nio.file.Path results = java.nio.file.Path.of("results");
            if (java.nio.file.Files.exists(results)) {
                try (var stream = java.nio.file.Files.newDirectoryStream(results, "checkpoint_*.bin")) {
                    for (var p : stream) java.nio.file.Files.deleteIfExists(p);
                }
            }
        } catch (Exception ignored) {}
    }

    public static void main(String[] args) {
        launch(args);
    }
}