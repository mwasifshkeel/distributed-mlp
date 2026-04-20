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
