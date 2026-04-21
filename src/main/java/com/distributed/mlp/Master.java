package com.distributed.mlp;

import com.distributed.mlp.protocol.MessageProtocol;
import com.distributed.mlp.protocol.MessageProtocol.ProtocolException;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.List;

/**
 * Master server entrypoint for distributed training. 
**/
public final class Master {
    private static final int DEFAULT_PORT = 9000;
    private static final int DEFAULT_WORKERS = 3;

    private final int port;
    private final int expectedWorkers;
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

    private static final class WorkerHandler implements Runnable {
        private final int workerId;
        private final Socket socket;

        private WorkerHandler(int workerId, Socket socket) {
            this.workerId = workerId;
            this.socket = socket;
        }

        @Override
        public void run() {
            try (Socket autoCloseSocket = socket;
                 DataInputStream in = new DataInputStream(autoCloseSocket.getInputStream())) {

                while (true) {
                    int[] header = MessageProtocol.readHeader(in);
                    int totalLength = header[0];
                    int type = header[1];
                    int payloadLength = totalLength - MessageProtocol.HEADER_SIZE;

                    if (payloadLength > 0) {
                        in.skipNBytes(payloadLength);
                    }

                    System.out.printf("Worker %d -> message type=0x%02X, payload=%dB%n", workerId, type, payloadLength);
                }
            } catch (EOFException | SocketException e) {
                System.out.printf("Worker %d disconnected.%n", workerId);
            } catch (ProtocolException e) {
                System.err.printf("Worker %d protocol error: %s%n", workerId, e.getMessage());
            } catch (IOException e) {
                System.err.printf("Worker %d I/O error: %s%n", workerId, e.getMessage());
            }
        }
    }
}
