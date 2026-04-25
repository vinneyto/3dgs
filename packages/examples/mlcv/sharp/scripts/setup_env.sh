#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MLCV_DIR="${ROOT_DIR}/mlcv"
VENV_DIR="${MLCV_DIR}/.env"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup] root: ${ROOT_DIR}"
echo "[setup] venv: ${VENV_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[setup] ERROR: ${PYTHON_BIN} not found. Install Python 3 first." >&2
  exit 1
fi

mkdir -p "${MLCV_DIR}"

if [ ! -f "${VENV_DIR}/pyvenv.cfg" ]; then
  echo "[setup] creating venv..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "[setup] venv already exists; reusing"
fi

VENV_PY="${VENV_DIR}/bin/python"

echo "[setup] upgrading pip..."
"${VENV_PY}" -m pip install --upgrade pip

echo "[setup] installing SHARP (official) + hf tooling..."
"${VENV_PY}" -m pip install \
  "huggingface-hub" \
  "git+https://github.com/apple/ml-sharp.git"

echo "[setup] freezing requirements..."
"${VENV_PY}" -m pip freeze > "${MLCV_DIR}/requirements.txt"

echo "[setup] done."
echo "[setup] check: ${VENV_DIR}/bin/sharp --help"

