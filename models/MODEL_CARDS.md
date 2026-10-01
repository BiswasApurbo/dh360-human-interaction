# Model Cards

This document tracks all pretrained machine learning models evaluated and utilized in the `dh360-human-interaction` vision pipeline.

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
python scripts/download_models.py --resource yunet
```
Target path: `models/weights/face_detection_yunet_2023mar.onnx` (ignored by git).

---

## 2. SFace FP32 Face Recognition

- **Model Name**: SFace FP32 Face Recognition (`face_recognition_sface_2021dec.onnx`)
- **Origin / Source Repository**: [OpenCV Model Zoo - SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- **Direct Download URL**: `https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx`
- **Original Paper**: Zhong et al., *"SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition"*, IEEE Transactions on Image Processing, 2021.
- **License**: Apache License 2.0
- **Version**: 2021-Dec Release
- **Format**: ONNX (`.onnx`)
- **File Size**: ~38.7 MB (38,696,353 bytes)
- **SHA-256 Checksum**:
  `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`

### Technical Specifications
- **Task**: Face feature embedding extraction from aligned face crops.
- **Input Representation**: Aligned 5-point canonical face image `(112, 112, 3)` BGR.
- **Output Representation**: 128-dimensional float32 feature vector.
- **Normalization Convention**: Standard L2 normalization ($||e||_2 = 1.0$).
- **Inference Runtime Engine**: OpenCV DNN / `cv2.FaceRecognizerSF`.

### Usage & Acquisition
```bash
python scripts/download_models.py --resource sface
```
Target path: `models/weights/face_recognition_sface_2021dec.onnx` (ignored by git).

---

## 3. SFace INT8 Quantized Face Recognition

- **Model Name**: SFace INT8 Quantized (`face_recognition_sface_2021dec_int8.onnx`)
- **Origin / Source Repository**: [OpenCV Model Zoo - SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- **Direct Download URL**: `https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx`
- **Original Paper**: Zhong et al., *"SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition"*, 2021.
- **License**: Apache License 2.0
- **Version**: 2021-Dec INT8 Quantized
- **Format**: ONNX (`.onnx`)
- **File Size**: ~9.9 MB (9,896,933 bytes)
- **SHA-256 Checksum**:
  `2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a`

### Technical Specifications
- **Task**: 8-bit quantized face feature embedding extraction.
- **Input Representation**: Aligned 5-point canonical face image `(112, 112, 3)` BGR.
- **Output Representation**: 128-dimensional float32 feature vector.
- **Normalization Convention**: Standard L2 normalization ($||e||_2 = 1.0$).
- **Inference Runtime Engine**: OpenCV DNN / `cv2.FaceRecognizerSF`.

### Usage & Acquisition
```bash
python scripts/download_models.py --resource sface_int8
```
Target path: `models/weights/face_recognition_sface_2021dec_int8.onnx` (ignored by git).

---

## 4. MobileFaceNet (Architectural Candidate Evaluation)

- **Model Name**: MobileFaceNet (Reference Architecture)
- **Original Paper**: Chen et al., *"MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices"*, Biometrics (CCBR), 2018.
- **Architectural Specs**: 0.99M parameters, ~4 MB model size, Global Depthwise Convolutional (GDConv) bottleneck.
- **Embedding Dimension**: 128-D or 512-D depending on loss head (ArcFace/InsightFace).
- **Input Representation**: `(112, 112, 3)` RGB/BGR.
- **License**: MIT / Apache 2.0 depending on upstream training repository.
- **Evaluation Status**: Architecturally viable for edge deployment; SFace was selected for primary integration due to official OpenCV C++/Python native binding (`cv2.FaceRecognizerSF`) requiring zero external dependencies or runtime translation.
