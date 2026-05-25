# Gsplat training Docker image (skeleton)

This folder contains a Docker image skeleton for running:

- Download images from Google Drive (rclone)
- COLMAP sparse reconstruction (sparse/0)
- gsplat training + PLY export
- Upload PLY back to Google Drive

The image exposes subcommands so each step can be tested independently.

## Files

- `Dockerfile` - builds the CUDA-based image (COLMAP + gsplat + rclone).
- `entrypoint.sh` - command dispatcher (`download`, `colmap-sparse`, `gsplat-ply`, `upload`, `all`).
- `rclone.conf.example` - example Google Drive OAuth config.

## Prerequisites (host)

For GPU use on Linux:

- NVIDIA driver installed (`nvidia-smi` works)
- `nvidia-container-toolkit` installed

For Mac or CPU-only tests:

- Run only `download` / `upload` or set `COLMAP_USE_GPU=0`
- Do not pass `--gpus all`

## Build

```bash
docker build -t gsplat-full:latest .
```

## rclone config (Google Drive)

Create config locally:

```bash
rclone config
```

It writes a file (example at `rclone.conf.example`).
Mount it into the container at:

```
/root/.config/rclone/rclone.conf
```

## Directory layout

```
RUN_DIR/
  images/   # downloaded photos
  colmap/   # COLMAP outputs (sparse/0)
  outputs/  # gsplat outputs (model.ply)
```

## Commands

### 1) Download images

```bash
docker run --rm \
  -v /path/to/rclone.conf:/root/.config/rclone/rclone.conf:ro \
  -v /data:/workspace \
  -e GDRIVE_INPUT_ID="FOLDER_ID" \
  -e RUN_DIR="/workspace/run_001" \
  gsplat-full:latest download
```

### 2) COLMAP sparse (GPU)

```bash
docker run --rm --gpus all \
  -v /data:/workspace \
  -e RUN_DIR="/workspace/run_001" \
  gsplat-full:latest colmap-sparse
```

CPU mode:

```bash
docker run --rm \
  -v /data:/workspace \
  -e RUN_DIR="/workspace/run_001" \
  -e COLMAP_USE_GPU=0 \
  gsplat-full:latest colmap-sparse
```

### 3) gsplat training + PLY

The exact gsplat command may differ by version. Pass it via `GSPLAT_CMD`:

```bash
docker run --rm --gpus all \
  -v /data:/workspace \
  -e RUN_DIR="/workspace/run_001" \
  -e GSPLAT_CMD='python3 examples/train.py --data "/workspace/run_001/colmap" --output "/workspace/run_001/outputs" --export_ply' \
  gsplat-full:latest gsplat-ply
```

If `/opt/gsplat/examples/train.py` exists, the entrypoint will use it by default
when `GSPLAT_CMD` is not provided.

### 4) Upload PLY

```bash
docker run --rm \
  -v /path/to/rclone.conf:/root/.config/rclone/rclone.conf:ro \
  -v /data:/workspace \
  -e RUN_DIR="/workspace/run_001" \
  -e GDRIVE_OUTPUT_DIR="gdrive:gsplat_outputs" \
  gsplat-full:latest upload
```

### 5) All-in-one

```bash
docker run --rm --gpus all \
  -v /path/to/rclone.conf:/root/.config/rclone/rclone.conf:ro \
  -v /data:/workspace \
  -e GDRIVE_INPUT_ID="FOLDER_ID" \
  -e GDRIVE_OUTPUT_DIR="gdrive:gsplat_outputs" \
  -e RUN_DIR="/workspace/run_001" \
  -e GSPLAT_CMD='python3 examples/train.py --data "/workspace/run_001/colmap" --output "/workspace/run_001/outputs" --export_ply' \
  gsplat-full:latest all
```

## Notes

- `GDRIVE_REMOTE` defaults to `gdrive` but can be overridden.
- `PLY_PATH` defaults to `RUN_DIR/outputs/model.ply`.
- This image is CUDA-based and intended for Linux/amd64.
