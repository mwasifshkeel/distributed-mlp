package com.distributed.mlp.gui;

import javafx.application.Platform;
import javafx.scene.Node;
import javafx.scene.control.TextArea;

/**
 * Shared scrolling log console used across all panels.
 */
public class LogConsole {

    private TextArea area;

    public Node build() {
        area = new TextArea();
        area.setEditable(false);
        area.setWrapText(false);
        area.getStyleClass().add("log-area");
        area.setPrefHeight(150);
        return area;
    }

    public void append(String line) {
        if (area == null) return;
        Platform.runLater(() -> {
            area.appendText(line + "\n");
            area.setScrollTop(Double.MAX_VALUE);
        });
    }

    public void clear() {
        if (area != null) Platform.runLater(() -> area.clear());
    }

    public java.util.function.Consumer<String> asConsumer() {
        return this::append;
    }
}
