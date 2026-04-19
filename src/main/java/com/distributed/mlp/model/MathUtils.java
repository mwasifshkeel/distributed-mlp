package com.distributed.mlp.model;

/**
 * Stateless math helpers used by the MLP implementation.
 */
public final class MathUtils {
    private MathUtils() {
    }

    public static double[] matVecMul(double[][] matrix, double[] vector) {
        if (matrix == null || vector == null) {
            throw new IllegalArgumentException("matrix and vector must not be null");
        }
        int rows = matrix.length;
        if (rows == 0) {
            return new double[0];
        }
        int cols = matrix[0].length;
        if (vector.length != cols) {
            throw new IllegalArgumentException("Dimension mismatch in matVecMul");
        }

        double[] out = new double[rows];
        for (int i = 0; i < rows; i++) {
            if (matrix[i].length != cols) {
                throw new IllegalArgumentException("Ragged matrix is not supported");
            }
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i][j] * vector[j];
            }
            out[i] = sum;
        }
        return out;
    }

    public static double[] addBias(double[] vector, double[] bias) {
        if (vector == null || bias == null) {
            throw new IllegalArgumentException("vector and bias must not be null");
        }
        if (vector.length != bias.length) {
            throw new IllegalArgumentException("Dimension mismatch in addBias");
        }
        double[] out = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            out[i] = vector[i] + bias[i];
        }
        return out;
    }

    public static double[] relu(double[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("vector must not be null");
        }
        double[] out = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            out[i] = Math.max(0.0, vector[i]);
        }
        return out;
    }

    public static double[] reluDeriv(double[] vector) {
        if (vector == null) {
            throw new IllegalArgumentException("vector must not be null");
        }
        double[] out = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            out[i] = vector[i] > 0.0 ? 1.0 : 0.0;
        }
        return out;
    }

    public static double[] softmax(double[] logits) {
        if (logits == null) {
            throw new IllegalArgumentException("logits must not be null");
        }
        if (logits.length == 0) {
            return new double[0];
        }

        double max = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }

        double sum = 0.0;
        double[] exp = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exp[i] = Math.exp(logits[i] - max);
            sum += exp[i];
        }

        double[] probs = new double[logits.length];
        if (sum == 0.0) {
            double uniform = 1.0 / logits.length;
            for (int i = 0; i < logits.length; i++) {
                probs[i] = uniform;
            }
            return probs;
        }

        for (int i = 0; i < logits.length; i++) {
            probs[i] = exp[i] / sum;
        }
        return probs;
    }

    public static double crossEntropyLoss(double[] probs, int label) {
        if (probs == null) {
            throw new IllegalArgumentException("probs must not be null");
        }
        if (label < 0 || label >= probs.length) {
            throw new IllegalArgumentException("label out of bounds");
        }
        double p = Math.max(probs[label], 1e-12);
        return -Math.log(p);
    }

    public static double[][] outerProduct(double[] a, double[] b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("vectors must not be null");
        }
        double[][] out = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                out[i][j] = a[i] * b[j];
            }
        }
        return out;
    }
}
