package com.distributed.mlp.model;

import java.util.Random;

public class MLPModel {
    public static final int INPUT_DIM = 3072;
    public static final int OUTPUT_DIM = 10;
    public static final int HIDDEN1_DIM = 512;
    public static final int HIDDEN2_DIM = 256;
    public static final int HIDDEN3_DIM = 128;

    // rows = output dim, cols = input dim (so matVecMul(w, x) works correctly)
    private final double[][] w1 = new double[HIDDEN1_DIM][INPUT_DIM];
    private final double[] b1 = new double[HIDDEN1_DIM];

    private final double[][] w2 = new double[HIDDEN2_DIM][HIDDEN1_DIM];
    private final double[] b2 = new double[HIDDEN2_DIM];

    private final double[][] w3 = new double[HIDDEN3_DIM][HIDDEN2_DIM];
    private final double[] b3 = new double[HIDDEN3_DIM];

    private final double[][] w4 = new double[OUTPUT_DIM][HIDDEN3_DIM];
    private final double[] b4 = new double[OUTPUT_DIM];

    public static int parameterCount() {
        return INPUT_DIM * HIDDEN1_DIM
                + HIDDEN1_DIM
                + HIDDEN1_DIM * HIDDEN2_DIM
                + HIDDEN2_DIM
                + HIDDEN2_DIM * HIDDEN3_DIM
                + HIDDEN3_DIM
                + HIDDEN3_DIM * OUTPUT_DIM
                + OUTPUT_DIM;
    }

    public void initXavier(long seed) {
        Random random = new Random(seed);
        initMatrixXavier(w1, INPUT_DIM, HIDDEN1_DIM, random);
        initBiasZero(b1);
        initMatrixXavier(w2, HIDDEN1_DIM, HIDDEN2_DIM, random);
        initBiasZero(b2);
        initMatrixXavier(w3, HIDDEN2_DIM, HIDDEN3_DIM, random);
        initBiasZero(b3);
        initMatrixXavier(w4, HIDDEN3_DIM, OUTPUT_DIM, random);
        initBiasZero(b4);
    }

    public double[][] getW1() { return w1; }
    public double[] getB1()   { return b1; }
    public double[][] getW2() { return w2; }
    public double[] getB2()   { return b2; }
    public double[][] getW3() { return w3; }
    public double[] getB3()   { return b3; }
    public double[][] getW4() { return w4; }
    public double[] getB4()   { return b4; }

    public void loadWeights(double[] flat) {
        if (flat == null) {
            throw new IllegalArgumentException("flat must not be null");
        }

        int expected = parameterCount();
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

        for (double[] row : w4)
            for (int j = 0; j < row.length; j++)
                row[j] = flat[idx++];

        for (int j = 0; j < b4.length; j++)
            b4[j] = flat[idx++];
    }

    public double[] toFlatWeights() {
        double[] flat = new double[parameterCount()];
        int idx = 0;
        idx = flatten2D(w1, flat, idx);
        idx = flatten1D(b1, flat, idx);
        idx = flatten2D(w2, flat, idx);
        idx = flatten1D(b2, flat, idx);
        idx = flatten2D(w3, flat, idx);
        idx = flatten1D(b3, flat, idx);
        idx = flatten2D(w4, flat, idx);
        flatten1D(b4, flat, idx);
        return flat;
    }

    public double[] forward(double[] input) {
        if (input == null) throw new IllegalArgumentException("input must not be null");
        if (input.length != INPUT_DIM)
            throw new IllegalArgumentException("Expected input length " + INPUT_DIM + " but got " + input.length);

        double[] z1 = MathUtils.addBias(MathUtils.matVecMul(w1, input), b1);
        double[] a1 = MathUtils.relu(z1);

        double[] z2 = MathUtils.addBias(MathUtils.matVecMul(w2, a1), b2);
        double[] a2 = MathUtils.relu(z2);

        double[] z3 = MathUtils.addBias(MathUtils.matVecMul(w3, a2), b3);
        double[] a3 = MathUtils.relu(z3);

        double[] logits = MathUtils.addBias(MathUtils.matVecMul(w4, a3), b4);
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
        double[] z3 = MathUtils.addBias(MathUtils.matVecMul(w3, a2), b3);
        double[] a3 = MathUtils.relu(z3);
        double[] logits = MathUtils.addBias(MathUtils.matVecMul(w4, a3), b4);
        double[] probs = MathUtils.softmax(logits);

        // Output layer gradient
        double[] dLogits = probs.clone();
        dLogits[label] -= 1.0;

        // Layer 4: w4 is [OUTPUT_DIM][HIDDEN3_DIM]
        double[][] dW4 = MathUtils.outerProduct(dLogits, a3);
        double[] db4 = dLogits.clone();

        double[] dA3 = matTVecMul(w4, dLogits);
        double[] dZ3 = hadamard(dA3, MathUtils.reluDeriv(z3));

        // Layer 3: w3 is [HIDDEN3_DIM][HIDDEN2_DIM]
        double[][] dW3 = MathUtils.outerProduct(dZ3, a2);
        double[] db3 = dZ3.clone();

        double[] dA2 = matTVecMul(w3, dZ3);
        double[] dZ2 = hadamard(dA2, MathUtils.reluDeriv(z2));

        // Layer 2: w2 is [HIDDEN2_DIM][HIDDEN1_DIM]
        double[][] dW2 = MathUtils.outerProduct(dZ2, a1);
        double[] db2 = dZ2.clone();

        double[] dA1 = matTVecMul(w2, dZ2);
        double[] dZ1 = hadamard(dA1, MathUtils.reluDeriv(z1));

        // Layer 1: w1 is [HIDDEN1_DIM][INPUT_DIM]
        double[][] dW1 = MathUtils.outerProduct(dZ1, input);
        double[] db1 = dZ1.clone();

        return new Gradient(dW1, db1, dW2, db2, dW3, db3, dW4, db4);
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

    private static int flatten2D(double[][] src, double[] dst, int idx) {
        for (double[] row : src) {
            for (double value : row) {
                dst[idx++] = value;
            }
        }
        return idx;
    }

    private static int flatten1D(double[] src, double[] dst, int idx) {
        for (double value : src) {
            dst[idx++] = value;
        }
        return idx;
    }

    public static final class Gradient {
        private final double[][] dW1;
        private final double[] db1;
        private final double[][] dW2;
        private final double[] db2;
        private final double[][] dW3;
        private final double[] db3;
        private final double[][] dW4;
        private final double[] db4;

        public Gradient(double[][] dW1, double[] db1, double[][] dW2,
                        double[] db2, double[][] dW3, double[] db3,
                        double[][] dW4, double[] db4) {
            this.dW1 = dW1; this.db1 = db1;
            this.dW2 = dW2; this.db2 = db2;
            this.dW3 = dW3; this.db3 = db3;
            this.dW4 = dW4; this.db4 = db4;
        }

        public double[][] getDW1() { return dW1; }
        public double[] getDb1()   { return db1; }
        public double[][] getDW2() { return dW2; }
        public double[] getDb2()   { return db2; }
        public double[][] getDW3() { return dW3; }
        public double[] getDb3()   { return db3; }
        public double[][] getDW4() { return dW4; }
        public double[] getDb4()   { return db4; }

        public double[] toFlatArray() {
            double[] flat = new double[parameterCount()];
            int idx = 0;
            idx = flatten2D(dW1, flat, idx);
            idx = flatten1D(db1, flat, idx);
            idx = flatten2D(dW2, flat, idx);
            idx = flatten1D(db2, flat, idx);
            idx = flatten2D(dW3, flat, idx);
            idx = flatten1D(db3, flat, idx);
            idx = flatten2D(dW4, flat, idx);
            flatten1D(db4, flat, idx);
            return flat;
        }
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