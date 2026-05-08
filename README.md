# Distributed MLP Training

**CS-347 Parallel & Distributed Computing** | **NUST SEECS**

## Overview
This repository contains the implementation of a distributed multi-layer perceptron (MLP) training pipeline for the **CIFAR-10** dataset. The system implements a master-worker (parameter-server) architecture with intra-node multi-threading, custom TCP binary communication, gradient compression, fault tolerance via checkpointing and heartbeat monitoring, and a full JavaFX GUI dashboard.

### Project Focus:
- Sequential baseline for correctness and reference timing
- Sync SGD (parallel baseline) with barrier synchronization
- Async distributed SGD over TCP sockets with parameter server
- Benchmarking: Strong scaling, weak scaling, Amdahl & Gustafson analysis
- Communication optimization using gradient compression (float32) and lazy weight pulls
- Fault tolerance: Checkpointing and automatic worker replacement
- Full JavaFX GUI dashboard with real-time monitoring

## Tech Stack
- Java 17+
- Maven 3.9+
- JavaFX for GUI dashboard
- TCP sockets (`java.net.Socket`)
- Concurrency primitives from `java.util.concurrent`

## Dataset Location
CIFAR-10 is automatically downloaded to:
```text
data/cifar-10-batches-bin/
```

## Repository Structure
```text
distributed-mlp/
├── pom.xml
├── README.md
├── run.sh                          # Main run script
├── data/
│   └── cifar-10-batches-bin/       # CIFAR-10 dataset (auto-downloaded)
├── logs/
│   ├── gui/                        # GUI inference logs
│   ├── master.log                  # Master process logs
│   ├── worker_0.log                # Worker 0 logs
│   ├── worker_1.log                # Worker 1 logs
│   └── worker_2.log                # Worker 2 logs
├── results/
│   ├── plots/                      # Generated benchmark plots
│   ├── checkpoint_*.bin            # Model checkpoints
│   ├── model_weights_*.bin         # Final model weights
│   └── *.csv                       # Benchmark data
└── src/main/java/com/distributed/mlp/
    ├── baseline/
    │   ├── SequentialBaseline.java     # Single-threaded reference
    │   └── SyncSGDBaseline.java        # Parallel baseline with barrier
    ├── bench/
    │   ├── BenchmarkRunner.java        # Run all benchmarks
    │   ├── BenchPlotter.java           # Generate plots from CSV
    │   ├── OptimizationBenchmark.java  # Test compression + lazy pull
    │   ├── ScalingExperiment.java      # Strong/weak scaling tests
    │   ├── SerializerBenchmark.java    # Serialization overhead
    │   ├── SpeedupAnalyzer.java        # Amdahl/Gustafson analysis
    │   └── ThreadScalingBenchmark.java # Intra-worker thread scaling
    ├── correctness/
    │   └── CorrectnessChecker.java     # Validate against sequential
    ├── data/
    │   └── DataLoader.java             # CIFAR-10 + sharding
    ├── gui/
    │   ├── MLPDashboard.java           # Main JavaFX dashboard
    │   ├── TrainingPanel.java          # Training controls
    │   ├── BenchmarkPanel.java         # Benchmark runner
    │   ├── ScalingPanel.java           # Scaling visualization
    │   ├── ChartsPanel.java            # Real-time charts
    │   ├── InferencePanel.java         # Model inference
    │   ├── CrashTestPanel.java         # Fault tolerance testing
    │   ├── ProcessManager.java         # Process lifecycle
    │   ├── LogConsole.java             # Log viewer
    │   └── DashboardLayout.java        # UI layout manager
    ├── model/
    │   ├── MathUtils.java              # Softmax, ReLU, etc.
    │   └── MLPModel.java               # Neural network (3072→128→64→10)
    ├── optimisation/
    │   └── GradientCompressor.java     # float32 compression
    ├── protocol/
    │   ├── MessageProtocol.java        # TCP message types (PULL, PUSH, PING, PONG, SHUTDOWN)
    │   └── WeightSerializer.java       # Binary serialization
    ├── util/
    │   └── MetricsLogger.java          # CSV timing output
    ├── Checkpoint.java                 # Model checkpointing (atomic writes, auto-prune)
    ├── Inference.java                  # Accuracy evaluation
    ├── Master.java                     # Parameter server with heartbeat monitoring
    ├── Worker.java                     # Gradient compute node with heartbeat sender
    └── WorkerReplacer.java             # Fault tolerance - automatic worker respawn
```

## Build Instructions

### Prerequisites
- Java 17 or higher
- Maven 3.6+

### Build Commands

```bash
# Clean previous builds
mvn clean

# Compile all sources
mvn compile

# Package JAR file
mvn package

# Run GUI dashboard (primary interface)
mvn javafx:run

# Or run the main class directly
java -cp target/classes com.distributed.mlp.gui.MLPDashboard
```

---

## Run Commands

### Quick Start Scripts

```bash
# Run all benchmarks (sequential, sync, scaling, optimization)
./run.sh
```

You can override defaults like WORKERS, INPUT_SIZE, EPOCHS, COMPUTE_THREADS, IO_THREADS, MASTER_PORT, WORKER_HEAP_MB, MASTER_HEAP_MB, PULL_EVERY, and COMPRESS_GRADIENTS when running ./run.sh, e.g. WORKERS=6 INPUT_SIZE=100000 EPOCHS=5 COMPUTE_THREADS=4 ./run.sh.

### 1. Sequential Baseline

```bash
# Run with default settings (1 epoch)
java -cp target/classes com.distributed.mlp.baseline.SequentialBaseline

# Run with custom parameters
java -cp target/classes com.distributed.mlp.baseline.SequentialBaseline --epochs 1 --samples 3000
```

### 2. Sync SGD (Parallel Baseline)

```bash
# Run with default settings (4 threads, batch size 128)
java -cp target/classes com.distributed.mlp.baseline.SyncSGDBaseline

# Run with custom parameters
java -cp target/classes com.distributed.mlp.baseline.SyncSGDBaseline --threads 4 --batchSize 128 --samples 3000 --epochs 1
```

### 3. Async Distributed SGD (Parameter Server)

#### Start Master First
```bash
# Basic master
java -cp target/classes com.distributed.mlp.Master

# With custom port and workers
java -cp target/classes com.distributed.mlp.Master 9000 3

# With checkpoint restore (automatic - just run master)
java -cp target/classes com.distributed.mlp.Master
```

#### Start Workers (in separate terminals)
```bash
# Worker 0
java -cp target/classes com.distributed.mlp.Worker --id 0 --master localhost --port 9000

# Worker 1
java -cp target/classes com.distributed.mlp.Worker --id 1 --master localhost --port 9000

# Worker 2
java -cp target/classes com.distributed.mlp.Worker --id 2 --master localhost --port 9000

# With custom thread count and batch size
java -cp target/classes com.distributed.mlp.Worker --id 0 --threads 4 --batchSize 128
```

### 4. GUI Dashboard

```bash
# Launch full GUI dashboard
mvn javafx:run

# Or directly
java -cp target/classes com.distributed.mlp.gui.MLPDashboard
```

The dashboard provides:
- **Training Panel**: Start/stop training, configure workers, threads, batch size
- **Benchmark Panel**: Run scaling experiments, optimization benchmarks
- **Scaling Panel**: Visualize strong/weak scaling results
- **Charts Panel**: Real-time loss and accuracy monitoring
- **Inference Panel**: Test trained models on sample images
- **Crash Test Panel**: Simulate worker failures to test fault tolerance

---

## Benchmark Commands

### Run All Benchmarks

```bash
# Complete benchmark suite (generates all CSV files and plots)
java -cp target/classes com.distributed.mlp.bench.BenchmarkRunner
```

### Individual Benchmarks

```bash
# Strong scaling (fixed problem size, vary workers)
java -cp target/classes com.distributed.mlp.bench.ScalingExperiment --strong --workers 1,2,4,8

# Weak scaling (problem size scales with workers)
java -cp target/classes com.distributed.mlp.bench.ScalingExperiment --weak --workers 1,2,4,8

# Thread scaling (vary threads per worker)
java -cp target/classes com.distributed.mlp.bench.ThreadScalingBenchmark --workers 3 --samples 10000 --threads 1,2,4,8

# Optimization benchmark (compression + lazy pull)
java -cp target/classes com.distributed.mlp.bench.OptimizationBenchmark --workers 3 --updates 1000

# Serialization benchmark
java -cp target/classes com.distributed.mlp.bench.SerializerBenchmark

# Speedup analysis (Amdahl & Gustafson)
java -cp target/classes com.distributed.mlp.bench.SpeedupAnalyzer --input results/strong_scaling.csv
```

### Generate Plots from Benchmark Data

```bash
# Generate all plots from CSV files in results/
java -cp target/classes com.distributed.mlp.bench.BenchPlotter
```

This generates:
- `results/plots/strong_speedup.png` - Speedup vs Workers
- `results/plots/weak_speedup.png` - Weak scaling performance
- `results/plots/thread_scaling_speedup.png` - Thread scaling
- `results/plots/amdahl_speedup.png` - Measured vs Theoretical
- `results/plots/gustafson_speedup.png` - Gustafson scaling
- `results/plots/optimisation_wall_sec.png` - Optimization comparison

### Correctness Validation

```bash
# Verify distributed results match sequential baseline
java -cp target/classes com.distributed.mlp.correctness.CorrectnessChecker
```

---

## Key Features Implemented

### Checkpoint System
- Master saves weights to `results/checkpoint_N.bin` every 5 gradient updates
- Atomic writes: write to `.tmp` first, then rename (no partial/corrupt checkpoints)
- On restart, automatically restores latest checkpoint and resumes from that update count
- Auto-prunes old checkpoints, keeping only 3 most recent
- Emergency checkpoint saved if all workers disconnect unexpectedly

### Protocol Message Types
| Type | Value | Direction | Purpose |
|------|-------|-----------|---------|
| PULL_REQUEST | 0x01 | Worker → Master | Request latest weights |
| WEIGHT_RESPONSE | 0x02 | Master → Worker | Deliver current weights |
| PUSH_GRADIENT | 0x03 | Worker → Master | Submit computed gradient |
| SHUTDOWN | 0x04 | Master → Worker | Signal training complete |
---

## Output Files

| File | Description |
|------|-------------|
| `results/strong_scaling.csv` | Strong scaling benchmark data |
| `results/weak_scaling.csv` | Weak scaling benchmark data |
| `results/thread_scaling.csv` | Thread scaling benchmark data |
| `results/optimisation_runs.csv` | Optimization benchmark results |
| `results/checkpoint_N.bin` | Periodic weight snapshot at update N (auto-pruned to 3 most recent) |
| `results/model_weights_*.bin` | Final saved model weights |
| `results/optimization_before_after.csv` | Time before and after optimization|
| `results/sequential_results.csv` | Sequential benchmark results  |
| `results/sync.csv` | SyncSGD benchmark results  |
| `results/thread_scaling.csv` | Parallelism benchmark results  |
| `results/speedup_table.txt` | Speedup on input size and workers  |
| `logs/master.log` | Master process logs |
| `logs/worker_*.log` | Worker process logs |
| `logs/gui/*.log` | GUI inference logs |
| `results/plots/*.png` | Generated benchmark plots |

---

## Troubleshooting

### CIFAR-10 Download Issues
```bash
# First run will auto-download; ensure internet connection
# Dataset cached in data/cifar-10-batches-bin/
```

### Port Already in Use
```bash
# Change default port (9000) when running master
java -cp target/classes com.distributed.mlp.Master 9001 3
java -cp target/classes com.distributed.mlp.Worker --port 9001
```

### Out of Memory
```bash
# Increase JVM heap
java -Xmx4g -cp target/classes com.distributed.mlp.bench.BenchmarkRunner
```

### GUI Not Launching
```bash
# Ensure JavaFX dependencies in pom.xml
mvn clean compile
mvn javafx:run

# Fallback: Run benchmarks without GUI
java -cp target/classes com.distributed.mlp.bench.BenchmarkRunner
```

### Checkpoint Recovery
```bash
# To restore from latest checkpoint, simply restart master
java -cp target/classes com.distributed.mlp.Master
# Master will auto-detect results/checkpoint_*.bin and resume
```


---

## Team

| Name | CMS ID |
|------|--------|
| Muhammad Wasif Shakeel | 456092 |
| Qasim Ahmed | 457282 |
| Muhammad Hanzala bin Rehan | 461738 |

**Course Instructor:** Dr. Fahad Satti  
**Course:** CS-347 Parallel & Distributed Computing  
**University:** NUST SEECS