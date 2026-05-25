#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-/workspace/run}"
IMAGES_DIR="${RUN_DIR}/images"
COLMAP_DIR="${RUN_DIR}/colmap"
OUTPUT_DIR="${RUN_DIR}/outputs"

GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
COLMAP_USE_GPU="${COLMAP_USE_GPU:-1}"

usage() {
  cat <<'EOF'
Usage: entrypoint.sh [download|colmap-sparse|gsplat-ply|upload|all]

Required env vars by command:
  download:
    GDRIVE_INPUT_ID   - Google Drive folder ID
  colmap-sparse:
    (uses RUN_DIR/images)
  gsplat-ply:
    GSPLAT_CMD (optional if /opt/gsplat/examples/train.py exists)
  upload:
    GDRIVE_OUTPUT_DIR - rclone destination, e.g. gdrive:gsplat_outputs

Common env vars:
  RUN_DIR            - base work dir (default: /workspace/run)
  GDRIVE_REMOTE      - rclone remote name (default: gdrive)
  COLMAP_USE_GPU     - 1 or 0 (default: 1)
  PLY_PATH           - path to PLY (default: RUN_DIR/outputs/model.ply)
EOF
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required env var: ${name}" >&2
    exit 1
  fi
}

cmd_download() {
  require_env GDRIVE_INPUT_ID
  mkdir -p "${IMAGES_DIR}"
  rclone copy "${GDRIVE_REMOTE}:${GDRIVE_INPUT_ID}" "${IMAGES_DIR}" --progress
}

cmd_colmap_sparse() {
  if [[ ! -d "${IMAGES_DIR}" ]]; then
    echo "Images dir not found: ${IMAGES_DIR}" >&2
    exit 1
  fi
  mkdir -p "${COLMAP_DIR}/sparse"

  colmap feature_extractor \
    --database_path "${COLMAP_DIR}/database.db" \
    --image_path "${IMAGES_DIR}" \
    --SiftExtraction.use_gpu "${COLMAP_USE_GPU}"

  colmap exhaustive_matcher \
    --database_path "${COLMAP_DIR}/database.db" \
    --SiftMatching.use_gpu "${COLMAP_USE_GPU}"

  colmap mapper \
    --database_path "${COLMAP_DIR}/database.db" \
    --image_path "${IMAGES_DIR}" \
    --output_path "${COLMAP_DIR}/sparse"
}

cmd_gsplat_ply() {
  mkdir -p "${OUTPUT_DIR}"

  if [[ -z "${GSPLAT_CMD:-}" ]]; then
    if [[ -f /opt/gsplat/examples/train.py ]]; then
      GSPLAT_CMD="python3 examples/train.py --data \"${COLMAP_DIR}\" --output \"${OUTPUT_DIR}\" --export_ply"
    else
      echo "GSPLAT_CMD is required (examples/train.py not found)." >&2
      exit 1
    fi
  fi

  (cd /opt/gsplat && bash -lc "${GSPLAT_CMD}")
}

cmd_upload() {
  require_env GDRIVE_OUTPUT_DIR
  local ply_path="${PLY_PATH:-${OUTPUT_DIR}/model.ply}"
  if [[ ! -f "${ply_path}" ]]; then
    echo "PLY not found: ${ply_path}" >&2
    exit 1
  fi
  rclone copy "${ply_path}" "${GDRIVE_OUTPUT_DIR}" --progress
}

cmd_all() {
  cmd_download
  cmd_colmap_sparse
  cmd_gsplat_ply
  cmd_upload
}

main() {
  local cmd="${1:-help}"
  case "${cmd}" in
    download) shift || true; cmd_download "$@" ;;
    colmap-sparse) shift || true; cmd_colmap_sparse "$@" ;;
    gsplat-ply) shift || true; cmd_gsplat_ply "$@" ;;
    upload) shift || true; cmd_upload "$@" ;;
    all) shift || true; cmd_all "$@" ;;
    help|--help|-h) usage ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
