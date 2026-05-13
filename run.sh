#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_ADDRESS="${1:-${HOST:-${SERVER_NAME:-0.0.0.0}}}"
PORT_NUMBER="${2:-${PORT:-${SERVER_PORT:-8080}}}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
PYTHON_BIN="${VENV_DIR}/bin/python"

cd "${PROJECT_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
  "${PROJECT_DIR}/install.sh"
fi

# Load environment for both interactive shells and systemd ExecStart.
if [ -f "${PROJECT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
  set +a
fi

# Positional arguments intentionally override .env for systemd templates and shell use.
export HOST="${SERVER_ADDRESS}"
export PORT="${PORT_NUMBER}"
export SERVER_NAME="${SERVER_ADDRESS}"
export SERVER_PORT="${PORT_NUMBER}"

# Safe accelerator defaults for CPU, NVIDIA H100, and NVIDIA DGX Spark-class systems.
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

UVICORN_ARGS=(app:app --host "${HOST}" --port "${PORT}")
if [ -n "${UVICORN_WORKERS:-}" ]; then
  UVICORN_ARGS+=(--workers "${UVICORN_WORKERS}")
fi
if [ -n "${UVICORN_LOG_LEVEL:-}" ]; then
  UVICORN_ARGS+=(--log-level "${UVICORN_LOG_LEVEL}")
fi

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exec "${PYTHON_BIN}" -m uvicorn "${UVICORN_ARGS[@]}"
else
  "${PYTHON_BIN}" -m uvicorn "${UVICORN_ARGS[@]}"
fi
