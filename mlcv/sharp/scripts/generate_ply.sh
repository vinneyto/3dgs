#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MLCV_DIR="${ROOT_DIR}/mlcv"
SHARP_DIR="${MLCV_DIR}/sharp"
VENV_DIR="${MLCV_DIR}/.env"

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/photo.jpg [optional_output_dir]" >&2
  echo "Example: $0 ${SHARP_DIR}/input/photo.jpg" >&2
  exit 1
fi

INPUT_PATH="$1"
OUT_DIR="${2:-${SHARP_DIR}/output}"
CKPT="${SHARP_DIR}/sharp_2572gikvuh.pt"

if [ ! -x "${VENV_DIR}/bin/sharp" ]; then
  echo "[generate] ERROR: SHARP CLI not found at ${VENV_DIR}/bin/sharp" >&2
  echo "[generate] Run: ${ROOT_DIR}/mlcv/sharp/scripts/setup_env.sh" >&2
  exit 1
fi

if [ ! -f "${INPUT_PATH}" ]; then
  echo "[generate] ERROR: input file not found: ${INPUT_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

if [ ! -f "${CKPT}" ]; then
  echo "[generate] WARNING: checkpoint not found at ${CKPT}" >&2
  echo "[generate] The SHARP CLI may try to auto-download on first run (see ml-sharp README)." >&2
fi

echo "[generate] input: ${INPUT_PATH}"
echo "[generate] out:   ${OUT_DIR}"
echo "[generate] ckpt:  ${CKPT}"

# Keep HuggingFace/torch caches inside the repo.
export HF_HOME="${MLCV_DIR}/.hf-cache"

"${VENV_DIR}/bin/sharp" predict \
  -i "${INPUT_PATH%/*}" \
  -o "${OUT_DIR}" \
  -c "${CKPT}"

echo "[generate] done. output directory: ${OUT_DIR}"

