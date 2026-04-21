package com.distributed.mlp;

import com.distributed.mlp.protocol.MessageProtocol;
import com.distributed.mlp.protocol.MessageProtocol.ProtocolException;
import com.distributed.mlp.protocol.WeightSerializer;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Master server entrypoint for distributed training. 
**/
public final class Master {
    private static final int DEFAULT_PORT = 9000;
    private static final int DEFAULT_WORKERS = 3;
    private static final double DEFAULT_LEARNING_RATE = 1e-3;
    private static final int TOTAL_PARAMETERS =
            com.distributed.mlp.model.MLPModel.INPUT_DIM * com.distributed.mlp.model.MLPModel.HIDDEN1_DIM
                    + com.distributed.mlp.model.MLPModel.HIDDEN1_DIM
                    + com.distributed.mlp.model.MLPModel.HIDDEN1_DIM * com.distributed.mlp.model.MLPModel.HIDDEN2_DIM
                    + com.distributed.mlp.model.MLPModel.HIDDEN2_DIM
                    + com.distributed.mlp.model.MLPModel.HIDDEN2_DIM * com.distributed.mlp.model.MLPModel.OUTPUT_DIM
                    + com.distributed.mlp.model.MLPModel.OUTPUT_DIM;

    private final int port;
    private final int expectedWorkers;
    private final double learningRate;
    private final double[] globalWeights;
    private final AtomicInteger totalUpdates;
    private final ReentrantLock weightLock;
    private final List<Thread> workerThreads = new ArrayList<>();

    public Master(int port, int expectedWorkers) {
        if (port <= 0 || port > 65535) {
            throw new IllegalArgumentException("port out of range: " + port);
        }
        if (expectedWorkers <= 0) {
            throw new IllegalArgumentException("expectedWorkers must be > 0");
        }
        this.port = port;
        this.expectedWorkers = expectedWorkers;
        this.learningRate = DEFAULT_LEARNING_RATE;
        this.globalWeights = new double[TOTAL_PARAMETERS];
        this.totalUpdates = new AtomicInteger(0);
        this.weightLock = new ReentrantLock();
    }

    public static void main(String[] args) {
        int port = args.length >= 1 ? Integer.parseInt(args[0]) : DEFAULT_PORT;
        int workers = args.length >= 2 ? Integer.parseInt(args[1]) : DEFAULT_WORKERS;

        Master master = new Master(port, workers);
        try {
            master.start();
        } catch (IOException e) {
            System.err.println("Master failed: " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    /**
     * Accepts worker sockets and starts one handler thread per worker.
     */
    public void start() throws IOException {
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.printf("Master listening on port %d, waiting for %d workers...%n", port, expectedWorkers);

            for (int workerId = 0; workerId < expectedWorkers; workerId++) {
                Socket workerSocket = serverSocket.accept();
                System.out.printf("Worker %d connected from %s%n", workerId, workerSocket.getRemoteSocketAddress());

                Thread handler = new Thread(
                        new WorkerHandler(workerId, workerSocket),
                        "master-worker-handler-" + workerId
                );
                handler.start();
                workerThreads.add(handler);
            }

            System.out.printf("All %d workers connected. Handlers are running.%n", expectedWorkers);

            for (Thread thread : workerThreads) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
    }

    /**
     * Returns a defensive snapshot of the current global weight vector.
     */
    public double[] snapshotGlobalWeights() {
        weightLock.lock();
        try {
            return Arrays.copyOf(globalWeights, globalWeights.length);
        } finally {
            weightLock.unlock();
        }
    }

    /**
     * Applies one asynchronous SGD update and increments the global update counter.
     */
    public int applyGradient(double[] gradient) {
        if (gradient == null) {
            throw new IllegalArgumentException("gradient must not be null");
        }
        if (gradient.length != globalWeights.length) {
            throw new IllegalArgumentException(
                    "Expected gradient length " + globalWeights.length + " but got " + gradient.length);
        }

        weightLock.lock();
        try {
            for (int i = 0; i < globalWeights.length; i++) {
                globalWeights[i] -= learningRate * gradient[i];
            }
        } finally {
            weightLock.unlock();
        }
        return totalUpdates.incrementAndGet();
    }

    public int getTotalUpdates() {
        return totalUpdates.get();
    }

    public int getGlobalWeightCount() {
        return globalWeights.length;
    }

    private final class WorkerHandler implements Runnable {
        private final int workerId;
        private final Socket socket;

        private WorkerHandler(int workerId, Socket socket) {
            this.workerId = workerId;
            this.socket = socket;
        }

        @Override
        public void run() {
            try (Socket autoCloseSocket = socket;
                 DataInputStream in = new DataInputStream(autoCloseSocket.getInputStream());
                 DataOutputStream out = new DataOutputStream(autoCloseSocket.getOutputStream())) {

                while (true) {
                    int[] header = MessageProtocol.readHeader(in);
                    int totalLength = header[0];
                    int type = header[1];
                    int payloadLength = totalLength - MessageProtocol.HEADER_SIZE;

                    switch (type) {
                        case MessageProtocol.PULL_REQUEST -> {
                            if (payloadLength > 0) {
                                in.skipNBytes(payloadLength);
                            }
                            handlePullRequest(out);
                        }
                        default -> {
                            if (payloadLength > 0) {
                                in.skipNBytes(payloadLength);
                            }
                            System.out.printf("Worker %d -> unsupported message type=0x%02X, payload=%dB%n",
                                    workerId,
                                    type,
                                    payloadLength);
                        }
                    }
                }
            } catch (EOFException | SocketException e) {
                System.out.printf("Worker %d disconnected.%n", workerId);
            } catch (ProtocolException e) {
                System.err.printf("Worker %d protocol error: %s%n", workerId, e.getMessage());
            } catch (IOException e) {
                System.err.printf("Worker %d I/O error: %s%n", workerId, e.getMessage());
            }
        }

        private void handlePullRequest(DataOutputStream out) throws IOException, ProtocolException {
            byte[] payload = WeightSerializer.toBytesDouble(snapshotGlobalWeights());
            MessageProtocol.writeHeader(out, MessageProtocol.WEIGHT_RESPONSE, payload.length);
            out.write(payload);
            out.flush();
            System.out.printf("Worker %d <- WEIGHT_RESPONSE sent (%d bytes)%n", workerId, payload.length);
        }
    }
}
