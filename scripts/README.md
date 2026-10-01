# Utility Scripts

This directory houses operational, diagnostic, and development automation scripts.

## Scripts Implemented (M1)

### `download_models.py`
Downloads official pretrained model weights with SHA-256 cryptographic checksum verification. Model binaries are saved locally to `models/weights/` (git-ignored).

```bash
# Download all models (default: YuNet face detection)
python scripts/download_models.py

# Download specific model
python scripts/download_models.py --model yunet
```

### `generate_test_video.py`
Generates a deterministic synthetic test video file (without real faces or biometric data) for reproducible headless testing and pipeline benchmarking.

```bash
# Generate a test video in data/test_video.mp4
python scripts/generate_test_video.py --output data/test_video.mp4 --frames 200 --width 640 --height 480
```
