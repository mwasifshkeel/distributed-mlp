package com.distributed.mlp.model;

import java.util.Random;

public class MLPModel {
    public static final int INPUT_DIM = 3072;
    public static final int OUTPUT_DIM = 10;
    public static final int HIDDEN1_DIM = 128;
    public static final int HIDDEN2_DIM = 64;

    // rows = output dim, cols = input dim (so matVecMul(w, x) works correctly)
    private final double[][] w1 = new double[HIDDEN1_DIM][INPUT_DIM];
    private final double[] b1 = new double[HIDDEN1_DIM];

    private final double[][] w2 = new double[HIDDEN2_DIM][HIDDEN1_DIM];
    private final double[] b2 = new double[HIDDEN2_DIM];

    private final double[][] w3 = new double[OUTPUT_DIM][HIDDEN2_DIM];
    private final double[] b3 = new double[OUTPUT_DIM];

    public void initXavier(long seed) {
        Random random = new Random(seed);
        initMatrixXavier(w1, INPUT_DIM, HIDDEN1_DIM, random);
        initBiasZero(b1);
        initMatrixXavier(w2, HIDDEN1_DIM, HIDDEN2_DIM, random);
        initBiasZero(b2);
        initMatrixXavier(w3, HIDDEN2_DIM, OUTPUT_DIM, random);
        initBiasZero(b3);
    }

    public double[][] getW1() { return w1; }
    public double[] getB1()   { return b1; }
    public double[][] getW2() { return w2; }
    public double[] getB2()   { return b2; }
    public double[][] getW3() { return w3; }
    public double[] getB3()   { return b3; }

    public void loadWeights(double[] flat) {
        if (flat == null) {
            throw new IllegalArgumentException("flat must not be null");
        }

        int expected = INPUT_DIM * HIDDEN1_DIM
                + HIDDEN1_DIM
                + HIDDEN1_DIM * HIDDEN2_DIM
                + HIDDEN2_DIM
                + HIDDEN2_DIM * OUTPUT_DIM
                + OUTPUT_DIM;
        if (flat.length != expected) {
            throw new IllegalArgumentException("Expected " + expected + " weights but got " + flat.length);
        }

        int idx = 0;

        for (double[] row : w1)
            for (int j = 0; j < row.length; j++)
                row[j] = flat[idx++];

        for (int j = 0; j < b1.length; j++)
            b1[j] = flat[idx++];

        for (double[] row : w2)
            for (int j = 0; j < row.length; j++)
                row[j] = flat[idx++];

        for (int j = 0; j < b2.length; j++)
            b2[j] = flat[idx++];

        for (double[] row : w3)
            for (int j = 0; j < row.length; j++)
                row[j] = flat[idx++];

        for (int j = 0; j < b3.length; j++)
            b3[j] = flat[idx++];
    }

    public double[] forward(double[] input) {
        if (input == null) throw new IllegalArgumentException("input must not be null");
        if (input.length != INPUT_DIM)
            throw new IllegalArgumentException("Expected input length " + INPUT_DIM + " but got " + input.length);

        double[] z1 = MathUtils.addBias(MathUtils.matVecMul(w1, input), b1);
        double[] a1 = MathUtils.relu(z1);

        double[] z2 = MathUtils.addBias(MathUtils.matVecMul(w2, a1), b2);
        double[] a2 = MathUtils.relu(z2);

        double[] logits = MathUtils.addBias(MathUtils.matVecMul(w3, a2), b3);
        return MathUtils.softmax(logits);
    }

    public Gradient backward(double[] input, int label) {
        if (input == null) throw new IllegalArgumentException("input must not be null");
        if (input.length != INPUT_DIM)
            throw new IllegalArgumentException("Expected input length " + INPUT_DIM + " but got " + input.length);
        if (label < 0 || label >= OUTPUT_DIM)
            throw new IllegalArgumentException("label out of bounds: " + label);

        // Forward pass to get activations
        double[] z1 = MathUtils.addBias(MathUtils.matVecMul(w1, input), b1);
        double[] a1 = MathUtils.relu(z1);
        double[] z2 = MathUtils.addBias(MathUtils.matVecMul(w2, a1), b2);
        double[] a2 = MathUtils.relu(z2);
        double[] logits = MathUtils.addBias(MathUtils.matVecMul(w3, a2), b3);
        double[] probs = MathUtils.softmax(logits);

        // Output layer gradient
        double[] dLogits = probs.clone();
        dLogits[label] -= 1.0;

        // Layer 3: w3 is [OUTPUT_DIM][HIDDEN2_DIM]
        // dW3[i][j] = dLogits[i] * a2[j]  → outer(dLogits, a2)
        double[][] dW3 = MathUtils.outerProduct(dLogits, a2);
        double[] db3 = dLogits.clone();

        // dA2 = w3^T · dLogits  → matTVecMul(w3, dLogits) where w3 is [OUTPUT_DIM][HIDDEN2_DIM]
        // matTVecMul multiplies w^T so result length = HIDDEN2_DIM ✓
        double[] dA2 = matTVecMul(w3, dLogits);
        double[] dZ2 = hadamard(dA2, MathUtils.reluDeriv(z2));

        // Layer 2: w2 is [HIDDEN2_DIM][HIDDEN1_DIM]
        double[][] dW2 = MathUtils.outerProduct(dZ2, a1);
        double[] db2 = dZ2.clone();

        double[] dA1 = matTVecMul(w2, dZ2);
        double[] dZ1 = hadamard(dA1, MathUtils.reluDeriv(z1));

        // Layer 1: w1 is [HIDDEN1_DIM][INPUT_DIM]
        double[][] dW1 = MathUtils.outerProduct(dZ1, input);
        double[] db1 = dZ1.clone();

        return new Gradient(dW1, db1, dW2, db2, dW3, db3);
    }

    // Transpose multiply: w^T · v, where w is [rows][cols], result is [cols]
    private static double[] matTVecMul(double[][] matrix, double[] vector) {
        if (matrix == null || vector == null)
            throw new IllegalArgumentException("matrix and vector must not be null");
        if (matrix.length == 0) return new double[0];
        if (matrix.length != vector.length)
            throw new IllegalArgumentException("Dimension mismatch in matTVecMul: matrix.length="
                    + matrix.length + " vector.length=" + vector.length);

        int cols = matrix[0].length;
        double[] out = new double[cols];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < cols; j++) {
                out[j] += matrix[i][j] * vector[i];
            }
        }
        return out;
    }

    private static double[] hadamard(double[] a, double[] b) {
        if (a.length != b.length)
            throw new IllegalArgumentException("Dimension mismatch in hadamard");
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i] * b[i];
        return out;
    }

    public static final class Gradient {
        private final double[][] dW1;
        private final double[] db1;
        private final double[][] dW2;
        private final double[] db2;
        private final double[][] dW3;
        private final double[] db3;

        public Gradient(double[][] dW1, double[] db1, double[][] dW2,
                        double[] db2, double[][] dW3, double[] db3) {
            this.dW1 = dW1; this.db1 = db1;
            this.dW2 = dW2; this.db2 = db2;
            this.dW3 = dW3; this.db3 = db3;
        }

        public double[][] getDW1() { return dW1; }
        public double[] getDb1()   { return db1; }
        public double[][] getDW2() { return dW2; }
        public double[] getDb2()   { return db2; }
        public double[][] getDW3() { return dW3; }
        public double[] getDb3()   { return db3; }
    }

    private static void initMatrixXavier(double[][] weights, int fanIn, int fanOut, Random random) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        for (double[] row : weights)
            for (int j = 0; j < row.length; j++)
                row[j] = (random.nextDouble() * 2.0 - 1.0) * limit;
    }

    private static void initBiasZero(double[] bias) {
        for (int i = 0; i < bias.length; i++) bias[i] = 0.0;
    }
}