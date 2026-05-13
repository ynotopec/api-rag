#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_VERSION="${UV_VERSION:-}"

cd "${PROJECT_DIR}"

mkdir -p "$(dirname "${VENV_DIR}")"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if ! "${VENV_DIR}/bin/uv" --version >/dev/null 2>&1; then
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel
  if [ -n "${UV_VERSION}" ]; then
    "${VENV_DIR}/bin/python" -m pip install --upgrade "uv==${UV_VERSION}"
  else
    "${VENV_DIR}/bin/python" -m pip install --upgrade uv
  fi
fi

"${VENV_DIR}/bin/uv" pip install --python "${VENV_DIR}/bin/python" --upgrade -r requirements.txt

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  chmod 600 .env
  echo "Created .env from .env.example; edit tokens and upstream settings before production use."
fi

cat <<MSG
Installed ${PROJECT_NAME}
  venv: ${VENV_DIR}
  python: ${VENV_DIR}/bin/python
Run:
  source run.sh 0.0.0.0 8080
MSG
