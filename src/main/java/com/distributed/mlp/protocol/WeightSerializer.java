package com.distributed.mlp.protocol;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;

/**
 * ByteBuffer-based serializers for model weights and gradients.
 */
public final class WeightSerializer {
    private static final int DOUBLE_BYTES = Double.BYTES;
    private static final int FLOAT_BYTES = Float.BYTES;

    private WeightSerializer() {
    }

    /**
     * Serializes a double array to bytes using a DoubleBuffer view.
     */
    public static byte[] toBytesDouble(double[] values) {
        if (values == null) {
            throw new IllegalArgumentException("values must not be null");
        }
        byte[] out = new byte[values.length * DOUBLE_BYTES];
        ByteBuffer.wrap(out).asDoubleBuffer().put(values);
        return out;
    }

    /**
     * Deserializes bytes into a double array.
     */
    public static double[] fromBytesDouble(byte[] bytes) {
        if (bytes == null) {
            throw new IllegalArgumentException("bytes must not be null");
        }
        if (bytes.length % DOUBLE_BYTES != 0) {
            throw new IllegalArgumentException("Invalid double payload size: " + bytes.length);
        }

        DoubleBuffer buffer = ByteBuffer.wrap(bytes).asDoubleBuffer();
        double[] out = new double[buffer.remaining()];
        buffer.get(out);
        return out;
    }

    /**
     * Serializes a double array into float32 bytes for bandwidth reduction.
     */
    public static byte[] toBytesFloat(double[] values) {
        if (values == null) {
            throw new IllegalArgumentException("values must not be null");
        }

        byte[] out = new byte[values.length * FLOAT_BYTES];
        FloatBuffer floatBuffer = ByteBuffer.wrap(out).asFloatBuffer();
        for (double value : values) {
            floatBuffer.put((float) value);
        }
        return out;
    }

    /**
     * Deserializes float32 payload bytes into a double array.
     */
    public static double[] fromBytesFloat(byte[] bytes) {
        if (bytes == null) {
            throw new IllegalArgumentException("bytes must not be null");
        }
        if (bytes.length % FLOAT_BYTES != 0) {
            throw new IllegalArgumentException("Invalid float payload size: " + bytes.length);
        }

        FloatBuffer floatBuffer = ByteBuffer.wrap(bytes).asFloatBuffer();
        double[] out = new double[floatBuffer.remaining()];
        for (int i = 0; i < out.length; i++) {
            out[i] = floatBuffer.get();
        }
        return out;
    }
}
