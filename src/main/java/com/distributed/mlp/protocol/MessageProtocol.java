package com.distributed.mlp.protocol;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Defines protocol codes and framing helpers for Master-Worker communication.
 */
public final class MessageProtocol {
    public static final int INT_SIZE = Integer.BYTES;
    public static final int HEADER_SIZE = INT_SIZE * 2;

    public static final int PULL_REQUEST = 0x01;
    public static final int WEIGHT_RESPONSE = 0x02;
    public static final int PUSH_GRADIENT = 0x03;
    public static final int SHUTDOWN = 0x04;

    private MessageProtocol() {
    }

    /**
     * Reads an 8-byte header and returns a tuple-like array: {totalLength, messageType}.
     */
    public static int[] readHeader(DataInputStream in) throws IOException, ProtocolException {
        if (in == null) {
            throw new IllegalArgumentException("input stream must not be null");
        }

        int totalLength = in.readInt();
        int type = in.readInt();

        if (totalLength < HEADER_SIZE) {
            throw new ProtocolException("Invalid header length: " + totalLength);
        }
        if (!isValidType(type)) {
            throw new ProtocolException("Bad message type magic: 0x" + Integer.toHexString(type));
        }
        return new int[]{totalLength, type};
    }

    /**
     * Writes an 8-byte header: totalLength(4B), type(4B).
     */
    public static void writeHeader(DataOutputStream out, int type, int payloadLength) throws IOException, ProtocolException {
        if (out == null) {
            throw new IllegalArgumentException("output stream must not be null");
        }
        if (payloadLength < 0) {
            throw new ProtocolException("Payload length must be non-negative: " + payloadLength);
        }
        if (!isValidType(type)) {
            throw new ProtocolException("Bad message type magic: 0x" + Integer.toHexString(type));
        }

        int totalLength = HEADER_SIZE + payloadLength;
        out.writeInt(totalLength);
        out.writeInt(type);
    }

    private static boolean isValidType(int type) {
        return type == PULL_REQUEST
                || type == WEIGHT_RESPONSE
                || type == PUSH_GRADIENT
                || type == SHUTDOWN;
    }

    /**
     * Checked exception used for framing/protocol violations.
     */
    public static final class ProtocolException extends Exception {
        public ProtocolException(String message) {
            super(message);
        }
    }
}