#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/spark/projects/vllm-install/spark-vllm-docker"
CONTAINER_NAME="vllm_nemo_ngc"
IMAGE="nvcr.io/nvidia/vllm:26.01-py3"
MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
PORT="8000"

echo "[vllm] repo: ${REPO_DIR}"
echo "[vllm] container: ${CONTAINER_NAME}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[vllm] docker not found"
  exit 1
fi

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[vllm] missing repo directory: ${REPO_DIR}"
  exit 1
fi

cd "${REPO_DIR}"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[vllm] starting container via launch-cluster.sh"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  ./launch-cluster.sh --solo -d --name "${CONTAINER_NAME}" \
    -t "${IMAGE}" \
    --apply-mod mods/use-ngc-vllm \
    --apply-mod mods/nemotron-nano \
    start
else
  echo "[vllm] container already running"
fi

echo "[vllm] starting server process inside container"
docker exec "${CONTAINER_NAME}" bash -lc "
  if [[ -f /tmp/nemotron-serve.pid ]]; then
    old_pid=\$(cat /tmp/nemotron-serve.pid 2>/dev/null || true)
    if [[ -n \"\${old_pid}\" ]] && kill -0 \"\${old_pid}\" 2>/dev/null; then
      kill \"\${old_pid}\" || true
      sleep 1
    fi
    rm -f /tmp/nemotron-serve.pid
  fi
  pids=\$(pgrep -f '^vllm serve ' || true)
  if [[ -n \"\${pids}\" ]]; then
    kill \${pids} || true
    sleep 1
  fi
"
docker exec -d "${CONTAINER_NAME}" bash -lc "
  export VLLM_USE_FLASHINFER_MOE_FP4=1
  export VLLM_FLASHINFER_MOE_BACKEND=throughput
  nohup vllm serve ${MODEL} \
    --max-model-len 4096 \
    --port ${PORT} --host 0.0.0.0 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --load-format safetensors \
    --gpu-memory-utilization 0.6 \
    --max-num-seqs 1 \
    --enforce-eager \
    >/tmp/nemotron-serve.log 2>&1 &
  echo \$! >/tmp/nemotron-serve.pid
"

echo "[vllm] waiting for health endpoint: http://127.0.0.1:${PORT}/health"
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[vllm] ready"
    echo "[vllm] test:"
    echo "curl -s http://127.0.0.1:${PORT}/v1/models | jq ."
    exit 0
  fi
  if (( i % 12 == 0 )); then
    echo "[vllm] still loading (attempt ${i}/240)"
  fi
  sleep 5
done

echo "[vllm] timeout waiting for server. Last logs:"
docker exec "${CONTAINER_NAME}" bash -lc "tail -n 160 /tmp/nemotron-serve.log || true"
exit 1
