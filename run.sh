#!/usr/bin/env bash
MASTER_PORT=${1:-9000}
WORKERS=${2:-3}
INPUT_SIZE=${3:-50000}
COMPUTE_THREADS=${4:-${MLP_COMPUTE_THREADS:-1}}

# CIFAR-10: 50000 samples, mini-batch 32
TOTAL_SAMPLES=${MLP_TOTAL_SAMPLES:-${INPUT_SIZE}}
MINI_BATCH=128
EPOCHS=${MLP_EPOCHS:-2}

# Apply max samples cap if set
if [[ -n "${MLP_MAX_SAMPLES:-}" ]]; then
    EFFECTIVE_SAMPLES=$(( TOTAL_SAMPLES < MLP_MAX_SAMPLES ? TOTAL_SAMPLES : MLP_MAX_SAMPLES ))
else
    EFFECTIVE_SAMPLES=${TOTAL_SAMPLES}
fi

# steps per worker = (effective_samples / workers / mini_batch) * epochs
STEPS_PER_WORKER=$(( (EFFECTIVE_SAMPLES / WORKERS / MINI_BATCH) * EPOCHS ))
# Ensure at least 1 step per worker
if [ ${STEPS_PER_WORKER} -lt 1 ]; then
    STEPS_PER_WORKER=1
fi
TARGET_UPDATES=$(( WORKERS * STEPS_PER_WORKER ))

WORKER_HEAP_MB=${MLP_WORKER_HEAP:-512}
MASTER_HEAP_MB=${MLP_MASTER_HEAP:-256}

JAVA_OPTS=()
if [[ "${MLP_COMPRESS_GRADIENTS:-true}" == "true" ]]; then
    JAVA_OPTS+=("-Dmlp.compressGradients=true")
fi
if [[ -n "${MLP_PULL_EVERY:-}" ]]; then
    JAVA_OPTS+=("-Dmlp.pullEvery=${MLP_PULL_EVERY}")
fi
if [[ -n "${COMPUTE_THREADS}" ]]; then
    JAVA_OPTS+=("-Dmlp.computeThreads=${COMPUTE_THREADS}")
fi
if [[ -n "${MLP_IO_THREADS:-}" ]]; then
    JAVA_OPTS+=("-Dmlp.ioThreads=${MLP_IO_THREADS}")
fi
if [[ -n "${MLP_MAX_SAMPLES:-}" ]]; then
    JAVA_OPTS+=("-Dmlp.maxSamples=${MLP_MAX_SAMPLES}")
fi

JAR="target/distributed-mlp-0.1.0-SNAPSHOT.jar"
if [ ! -f "$JAR" ]; then
    echo "Building JAR..."
    mvn package -DskipTests -q
fi

mkdir -p logs

echo "========================================="
echo "Training Configuration:"
echo "========================================="
echo "  Workers         : ${WORKERS}"
echo "  Input size      : ${INPUT_SIZE}"
echo "  Effective size  : ${EFFECTIVE_SAMPLES}"
echo "  Epochs          : ${EPOCHS}"
echo "  Mini-batch      : ${MINI_BATCH}"
echo "  Compute threads : ${COMPUTE_THREADS}"
echo "  Steps/worker    : ${STEPS_PER_WORKER}"
echo "  Target updates  : ${TARGET_UPDATES}"
echo "  Master heap     : ${MASTER_HEAP_MB} MB"
echo "  Worker heap     : ${WORKER_HEAP_MB} MB"
echo "  Compress grads  : ${MLP_COMPRESS_GRADIENTS:-true}"
echo "========================================="

echo "Starting Master on port ${MASTER_PORT} ..."
java "${JAVA_OPTS[@]}" -Xmx${MASTER_HEAP_MB}m -cp target/classes com.distributed.mlp.Master \
    "${MASTER_PORT}" "${WORKERS}" "${STEPS_PER_WORKER}" 42 \
    > logs/master.log 2>&1 &
MASTER_PID=$!

sleep 2

for i in $(seq 0 $(( WORKERS - 1 ))); do
    echo "Starting Worker ${i} ..."
    java "${JAVA_OPTS[@]}" -Xmx${WORKER_HEAP_MB}m -cp target/classes com.distributed.mlp.Worker \
        127.0.0.1 "${MASTER_PORT}" "${i}" "${WORKERS}" "${STEPS_PER_WORKER}" 42 \
        > logs/worker_${i}.log 2>&1 &
done

echo ""
echo "All processes launched. Waiting for completion..."
echo "Logs are in ./logs/"
echo "Press Ctrl+C to stop early."
echo ""

trap 'echo "Stopping all processes..."; kill $(jobs -p) 2>/dev/null' INT TERM

# Wait for master to complete
wait ${MASTER_PID} 2>/dev/null
MASTER_EXIT=$?

# Give workers a moment to shutdown gracefully
sleep 10

# Kill any remaining workers
kill $(jobs -p) 2>/dev/null

echo ""
echo "========================================="
echo "Training complete (Master exit code: ${MASTER_EXIT})"
echo "========================================="