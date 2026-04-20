package com.distributed.mlp.model;

import java.util.Random;

/**
 * MLP parameter container with Xavier initialization.
 */
public class MLPModel {
    public static final int INPUT_DIM = 49152;
    public static final int HIDDEN1_DIM = 512;
    public static final int HIDDEN2_DIM = 256;
    public static final int OUTPUT_DIM = 101;

    private final double[][] w1 = new double[INPUT_DIM][HIDDEN1_DIM];
    private final double[] b1 = new double[HIDDEN1_DIM];

    private final double[][] w2 = new double[HIDDEN1_DIM][HIDDEN2_DIM];
    private final double[] b2 = new double[HIDDEN2_DIM];

    private final double[][] w3 = new double[HIDDEN2_DIM][OUTPUT_DIM];
    private final double[] b3 = new double[OUTPUT_DIM];

    /**
     * Xavier initializes all weight matrices and zero-initializes biases.
     */
    public void initXavier(long seed) {
        Random random = new Random(seed);

        initMatrixXavier(w1, INPUT_DIM, HIDDEN1_DIM, random);
        initBiasZero(b1);

        initMatrixXavier(w2, HIDDEN1_DIM, HIDDEN2_DIM, random);
        initBiasZero(b2);

        initMatrixXavier(w3, HIDDEN2_DIM, OUTPUT_DIM, random);
        initBiasZero(b3);
    }

    public double[][] getW1() {
        return w1;
    }

    public double[] getB1() {
        return b1;
    }

    public double[][] getW2() {
        return w2;
    }

    public double[] getB2() {
        return b2;
    }

    public double[][] getW3() {
        return w3;
    }

    public double[] getB3() {
        return b3;
    }

    /**
     * Runs forward propagation: affine + ReLU + affine + ReLU + affine + softmax.
     */
    public double[] forward(double[] input) {
        if (input == null) {
            throw new IllegalArgumentException("input must not be null");
        }
        if (input.length != INPUT_DIM) {
            throw new IllegalArgumentException(
                    "Expected input length " + INPUT_DIM + " but got " + input.length);
        }

        double[] z1 = MathUtils.addBias(MathUtils.matVecMul(w1, input), b1);
        double[] a1 = MathUtils.relu(z1);

        double[] z2 = MathUtils.addBias(MathUtils.matVecMul(w2, a1), b2);
        double[] a2 = MathUtils.relu(z2);

        double[] logits = MathUtils.addBias(MathUtils.matVecMul(w3, a2), b3);
        return MathUtils.softmax(logits);
    }

    /**
     * Computes gradients for one sample using cross-entropy with softmax output.
     */
    public Gradient backward(double[] input, int label) {
        if (input == null) {
            throw new IllegalArgumentException("input must not be null");
        }
        if (input.length != INPUT_DIM) {
            throw new IllegalArgumentException(
                    "Expected input length " + INPUT_DIM + " but got " + input.length);
        }
        if (label < 0 || label >= OUTPUT_DIM) {
            throw new IllegalArgumentException("label out of bounds: " + label);
        }

        double[] z1 = MathUtils.addBias(MathUtils.matVecMul(w1, input), b1);
        double[] a1 = MathUtils.relu(z1);

        double[] z2 = MathUtils.addBias(MathUtils.matVecMul(w2, a1), b2);
        double[] a2 = MathUtils.relu(z2);

        double[] logits = MathUtils.addBias(MathUtils.matVecMul(w3, a2), b3);
        double[] probs = MathUtils.softmax(logits);

        double[] dLogits = probs.clone();
        dLogits[label] -= 1.0;

        double[][] dW3 = MathUtils.outerProduct(a2, dLogits);
        double[] db3 = dLogits.clone();

        double[] dA2 = matTVecMul(w3, dLogits);
        double[] dZ2 = hadamard(dA2, MathUtils.reluDeriv(z2));

        double[][] dW2 = MathUtils.outerProduct(a1, dZ2);
        double[] db2 = dZ2.clone();

        double[] dA1 = matTVecMul(w2, dZ2);
        double[] dZ1 = hadamard(dA1, MathUtils.reluDeriv(z1));

        double[][] dW1 = MathUtils.outerProduct(input, dZ1);
        double[] db1 = dZ1.clone();

        return new Gradient(dW1, db1, dW2, db2, dW3, db3);
    }

    private static double[] matTVecMul(double[][] matrix, double[] vector) {
        if (matrix == null || vector == null) {
            throw new IllegalArgumentException("matrix and vector must not be null");
        }
        if (matrix.length == 0) {
            return new double[0];
        }
        if (matrix[0].length != vector.length) {
            throw new IllegalArgumentException("Dimension mismatch in matTVecMul");
        }

        int rows = matrix.length;
        int cols = matrix[0].length;
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

    private static double[] hadamard(double[] a, double[] b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("vectors must not be null");
        }
        if (a.length != b.length) {
            throw new IllegalArgumentException("Dimension mismatch in hadamard");
        }
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] * b[i];
        }
        return out;
    }

    /**
     * Per-layer gradients for one backward pass.
     */
    public static final class Gradient {
        private final double[][] dW1;
        private final double[] db1;
        private final double[][] dW2;
        private final double[] db2;
        private final double[][] dW3;
        private final double[] db3;

        public Gradient(
                double[][] dW1,
                double[] db1,
                double[][] dW2,
                double[] db2,
                double[][] dW3,
                double[] db3) {
            this.dW1 = dW1;
            this.db1 = db1;
            this.dW2 = dW2;
            this.db2 = db2;
            this.dW3 = dW3;
            this.db3 = db3;
        }

        public double[][] getDW1() {
            return dW1;
        }

        public double[] getDb1() {
            return db1;
        }

        public double[][] getDW2() {
            return dW2;
        }

        public double[] getDb2() {
            return db2;
        }

        public double[][] getDW3() {
            return dW3;
        }

        public double[] getDb3() {
            return db3;
        }
    }

    private static void initMatrixXavier(double[][] weights, int fanIn, int fanOut, Random random) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = (random.nextDouble() * 2.0 - 1.0) * limit;
            }
        }
    }

    private static void initBiasZero(double[] bias) {
        for (int i = 0; i < bias.length; i++) {
            bias[i] = 0.0;
        }
    }
}
