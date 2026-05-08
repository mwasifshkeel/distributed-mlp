#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Configuration (GUI can override via env)
# =========================================
MASTER_PORT=${MASTER_PORT:-9000}
WORKERS=${WORKERS:-3}
INPUT_SIZE=${INPUT_SIZE:-50000}
COMPUTE_THREADS=${COMPUTE_THREADS:-1}
EPOCHS=${EPOCHS:-2}

MINI_BATCH=128
TOTAL_SAMPLES=${INPUT_SIZE}

COMPRESS_GRADIENTS=${COMPRESS_GRADIENTS:-true}
PULL_EVERY=${PULL_EVERY:-10}

WORKER_HEAP_MB=${WORKER_HEAP_MB:-512}
MASTER_HEAP_MB=${MASTER_HEAP_MB:-256}
IO_THREADS=${IO_THREADS:-1}

# =========================================
# Derived workload
# =========================================
STEPS_PER_WORKER=$(( (TOTAL_SAMPLES / WORKERS / MINI_BATCH) * EPOCHS ))

if [ "$STEPS_PER_WORKER" -lt 1 ]; then
    STEPS_PER_WORKER=1
fi

TARGET_UPDATES=$(( WORKERS * STEPS_PER_WORKER ))

# =========================================
# JVM options
# =========================================
JAVA_OPTS=(
    "-Dmlp.compressGradients=${COMPRESS_GRADIENTS}"
    "-Dmlp.pullEvery=${PULL_EVERY}"
    "-Dmlp.computeThreads=${COMPUTE_THREADS}"
    "-Dmlp.ioThreads=${IO_THREADS}"
)

# =========================================
# Build
# =========================================
JAR="target/distributed-mlp-0.1.0-SNAPSHOT.jar"

if [ ! -f "$JAR" ]; then
    echo "Building project..."
    mvn package -DskipTests -q
fi

mkdir -p logs

# =========================================
# Print config
# =========================================
echo "========================================="
echo "Training Configuration"
echo "========================================="
echo "Master Port      : $MASTER_PORT"
echo "Workers          : $WORKERS"
echo "Input Size       : $INPUT_SIZE"
echo "Epochs           : $EPOCHS"
echo "Mini Batch       : $MINI_BATCH"
echo "Compute Threads  : $COMPUTE_THREADS"
echo "Pull Every       : $PULL_EVERY"
echo "Compress Grad    : $COMPRESS_GRADIENTS"
echo "Steps / Worker   : $STEPS_PER_WORKER"
echo "Target Updates   : $TARGET_UPDATES"
echo "========================================="

# =========================================
# Start Master
# =========================================
echo "Starting Master..."

java "${JAVA_OPTS[@]}" \
    -Xmx${MASTER_HEAP_MB}m \
    -cp target/classes \
    com.distributed.mlp.Master \
    "$MASTER_PORT" "$WORKERS" "$STEPS_PER_WORKER" 42 \
    > logs/master.log 2>&1 &

MASTER_PID=$!

sleep 2

# =========================================
# Start Workers
# =========================================
for i in $(seq 0 $((WORKERS - 1))); do
    echo "Starting Worker $i..."

    java "${JAVA_OPTS[@]}" \
        -Xmx${WORKER_HEAP_MB}m \
        -cp target/classes \
        com.distributed.mlp.Worker \
        127.0.0.1 "$MASTER_PORT" "$i" "$WORKERS" "$STEPS_PER_WORKER" 42 \
        > logs/worker_${i}.log 2>&1 &
done

echo ""
echo "Training started..."
echo ""

trap 'echo "Stopping..."; kill $(jobs -p) 2>/dev/null' INT TERM

wait ${MASTER_PID}
MASTER_EXIT=$?

sleep 5
kill $(jobs -p) 2>/dev/null || true

echo ""
echo "========================================="
echo "Training Complete"
echo "Exit Code: $MASTER_EXIT"
echo "========================================="