# Model Cards

This document tracks all pretrained machine learning models utilized in the `dh360-human-interaction` vision pipeline.

---

## 1. YuNet Face Detection

- **Model Name**: YuNet Face Detection (`face_detection_yunet_2023mar.onnx`)
- **Origin / Source Repository**: [OpenCV Model Zoo - YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Direct Download URL**: `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
- **Original Paper / Project**: [ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection)
- **License**: Apache License 2.0
- **Version**: 2023-Mar Release
- **Format**: ONNX (`.onnx`)
- **File Size**: ~232 KB (232,589 bytes)
- **SHA-256 Checksum**:
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`

### Technical Specifications
- **Task**: Real-time 2D face detection and 5-point facial landmark localization.
- **Input Representation**: BGR color image tensor `(1, 3, H, W)`.
- **Output Representation**: For each detected face, a 14-element feature vector:
  - Bounding Box: `[x_min, y_min, width, height]` (pixels)
  - Landmarks: `[right_eye_x, right_eye_y, left_eye_x, left_eye_y, nose_x, nose_y, right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y]`
  - Detection Confidence Score: `[score]` in range `[0.0, 1.0]`
- **Inference Runtime Engine**: OpenCV DNN / `cv2.FaceDetectorYN`.

### Usage & Acquisition
```bash
python scripts/download_models.py --model yunet
```
Target path: `models/weights/face_detection_yunet_2023mar.onnx` (ignored by git).
