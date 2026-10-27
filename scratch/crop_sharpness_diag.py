import os, sys, time, importlib.util, pathlib
import cv2
import numpy as np

# Ensure repo root is on PYTHONPATH for relative imports (src package)
repo_root = pathlib.Path(__file__).resolve().parents[1]
src_path = repo_root / 'src'
if src_path.is_dir():
    sys.path.insert(0, str(src_path))

# Stub tracker module to avoid circular import when loading aligner
stub_dir = repo_root / 'scratch'
stub_dir.mkdir(parents=True, exist_ok=True)
tracker_stub_path = stub_dir / 'tracker_stub.py'
if not tracker_stub_path.exists():
    tracker_stub_path.write_text('class TrackedFace: pass')
# Insert stub directory first in sys.path and create fake module
sys.path.insert(0, str(stub_dir))
import types
tracker_mod = types.ModuleType('dh360_hi.vision.tracker')
exec(open(tracker_stub_path).read(), tracker_mod.__dict__)
sys.modules['dh360_hi.vision.tracker'] = tracker_mod

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load core modules directly (avoiding package __init__ side‑effects)
frame_mod = load_module('dh360_hi.vision.frame', str(repo_root / 'src' / 'dh360_hi' / 'vision' / 'frame.py'))
Frame = frame_mod.Frame

det_mod = load_module('dh360_hi.vision.detector', str(repo_root / 'src' / 'dh360_hi' / 'vision' / 'detector.py'))
YuNetFaceDetector = det_mod.YuNetFaceDetector
FaceDetection = det_mod.FaceDetection
FaceLandmarks = det_mod.FaceLandmarks

align_mod = load_module('dh360_hi.vision.aligner', str(repo_root / 'src' / 'dh360_hi' / 'vision' / 'aligner.py'))
FaceAligner = align_mod.FaceAligner

# Model path – use provided default if env var missing
model_path = os.getenv('DH360_HI_DETECTOR_MODEL_PATH', 'models/weights/face_detection_yunet_2023mar.onnx')
if not os.path.isfile(model_path):
    raise RuntimeError(f'Model file not found: {model_path}')

detector = YuNetFaceDetector(model_path=model_path)

# Webcam index – default 0
cam_idx = int(os.getenv('DH360_HI_CAMERA_INDEX', '0'))
cap = cv2.VideoCapture(cam_idx)
if not cap.isOpened():
    raise RuntimeError(f'Failed to open webcam {cam_idx}')

print('Collecting diagnostics (≈10 successful detections)')
records = []
while len(records) < 10:
    ret, img = cap.read()
    if not ret:
        continue
    frame = Frame(image=img, frame_id=len(records), timestamp=time.time())
    det_res = detector.detect(frame)
    if not det_res.detections:
        continue
    detection = det_res.detections[0]
    x, y, w, h = detection.bbox
    img_h, img_w = img.shape[:2]
    # Raw crop (exact detector bbox)
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(img_w, int(np.ceil(x + w)))
    y2 = min(img_h, int(np.ceil(y + h)))
    raw_crop = img[y1:y2, x1:x2]
    raw_gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY) if raw_crop.ndim == 3 else raw_crop
    raw_sharp = float(cv2.Laplacian(raw_gray, cv2.CV_64F).var())
    # Padded crop (10% of w/h on each side)
    pad_w = 0.1 * w
    pad_h = 0.1 * h
    x1p = max(0, int(np.floor(x - pad_w)))
    y1p = max(0, int(np.floor(y - pad_h)))
    x2p = min(img_w, int(np.ceil(x + w + pad_w)))
    y2p = min(img_h, int(np.ceil(y + h + pad_h)))
    pad_crop = img[y1p:y2p, x1p:x2p]
    pad_gray = cv2.cvtColor(pad_crop, cv2.COLOR_BGR2GRAY) if pad_crop.ndim == 3 else pad_crop
    pad_sharp = float(cv2.Laplacian(pad_gray, cv2.CV_64F).var())
    # Aligned 112x112 crop via FaceAligner
    aligner = FaceAligner(output_size=(112, 112), fallback_to_bbox=True)
    align_res = aligner.align(frame, detection, detection.landmarks)
    aligned = align_res.aligned_face
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
    aligned_sharp = float(cv2.Laplacian(aligned_gray, cv2.CV_64F).var())
    records.append({
        'bbox_w': w,
        'bbox_h': h,
        'raw_sharp': raw_sharp,
        'pad_sharp': pad_sharp,
        'aligned_sharp': aligned_sharp,
    })
    time.sleep(0.2)

cap.release()

# Print summary statistics
print('\n--- Summary (per detection) ---')
for i, r in enumerate(records, 1):
    print(f"#{i:02d}: bbox={r['bbox_w']:.1f}x{r['bbox_h']:.1f}  raw={r['raw_sharp']:.1f}  pad={r['pad_sharp']:.1f}  aligned={r['aligned_sharp']:.1f}")

import statistics
raw_vals = [r['raw_sharp'] for r in records]
pad_vals = [r['pad_sharp'] for r in records]
aligned_vals = [r['aligned_sharp'] for r in records]
print('\n--- Distributions ---')
for name, vals in [('Raw', raw_vals), ('Padded', pad_vals), ('Aligned', aligned_vals)]:
    print(f"{name}: min={min(vals):.1f} max={max(vals):.1f} mean={statistics.mean(vals):.1f} stdev={statistics.stdev(vals):.1f}")
print('Done')
