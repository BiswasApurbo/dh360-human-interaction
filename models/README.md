# Pretrained Models Directory

This directory stores configuration metadata and local storage locations for pretrained computer vision models.

## Policy & Security Rules
- **Pretrained Only**: We utilize lightweight, standard pretrained models for face detection and embedding extraction; no models are trained from scratch.
- **No Binaries in Git**: Model weight files (`*.onnx`, `*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `*.engine`, `*.weights`) are ignored by version control.
- **Biometric Protection**: Raw model weights and test biometric embeddings must not be committed to repository history.
- Model artifacts should be downloaded via scripts with explicit SHA256 checksum verification.
