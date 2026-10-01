# Test Fixtures & Validation Samples

This directory documents test fixtures used for deterministic, non-identifying computer vision tests and functional validation benchmarks.

---

## 1. Sample Face Image (`sample_face.jpg`)

- **File**: `sample_face.jpg`
- **Origin / Source**: [OpenCV Sample Data](https://github.com/opencv/opencv/blob/4.x/samples/data/lena.jpg)
- **License**: BSD 3-Clause License (OpenCV Project)
- **Dimensions**: 512 x 512 pixels (RGB / BGR)
- **File Size**: 91,814 bytes (~91.8 KB)
- **SHA-256 Checksum**:
  `7de7ed51a1594fff247f4cae2301eceacf5313d6011e37b4a4c8733f7bb72c07`
- **Purpose**: A standard, non-identifying reference fixture used to verify face detector functionality, bounding box coordinates, and landmark extraction.
- **Privacy & Compliance**: Contains no private, personal, or clinical data.

---

## 2. Dynamic Video Generation

For multi-frame functional validation, `scripts/generate_test_video.py` can generate an animated multi-frame video stream (`data/face_test_video.mp4`, git-ignored) using `sample_face.jpg` to test continuous tracking and detection pipelines across 100+ frames with motion and illumination variation.
