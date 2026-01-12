## SHARP → PLY (single photo) quickstart

This folder contains a minimal setup to generate `.ply` (3D Gaussian Splats) from one or more photos using Apple's SHARP model.

### 0) (Optional) Hugging Face token for faster downloads

Get a token with **Read** scope at:

- `https://huggingface.co/settings/tokens`

Then in your shell:

```bash
export HF_TOKEN="hf_..."
```

### 1) Create/update the Python environment (venv)

From the `3dgs` repo root:

```bash
./mlcv/sharp/scripts/setup_env.sh
```

This will create/update:

- `mlcv/.env/` (Python venv)
- `mlcv/requirements.txt` (pinned via `pip freeze`)

### 2) Put photos into the input folder

Put your images here:

- `mlcv/sharp/input/`

Supported formats: `jpg/jpeg/png`.

### 3) Run inference to produce `.ply`

Generate `.ply` for a single photo:

```bash
./mlcv/sharp/scripts/generate_ply.sh mlcv/sharp/input/photo.jpg
```

Outputs:

- `mlcv/sharp/output/<photo_basename>.ply`

### Notes

- The script forces `HF_HOME=mlcv/.hf-cache` so caches stay inside the repo.
- Device selection is handled by PyTorch (on macOS you'll typically see `mps`).
