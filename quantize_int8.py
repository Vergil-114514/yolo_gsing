"""
INT8 quantization of YOLOv8n OpenVINO model using NNCF.
Uses only training set images for calibration (not valid/test).
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import cv2
import openvino as ov
import openvino.runtime.op as real_op

# Monkey-patch: NNCF expects old OpenVINO API (ov.Node, openvino.op)
# which moved in OV 2024.x to ov.runtime.Node, openvino.runtime.op
import sys
sys.modules['openvino.op'] = real_op
ov.op = real_op
ov.Node = ov.runtime.Node

import nncf

BASE = Path(r"e:/xwechat_files/wxid_el79rox3ahjk12_09c9/msg/file/2026-05/yolo_gsing")
RUNS = BASE / "runs" / "cube_yolov8n_n100"
FP32_MODEL = RUNS / "openvino_fp32" / "best.xml"
INT8_OUTPUT = RUNS / "openvino_int8"
CALIBRATION_IMAGES = BASE / "Cube.yolov8" / "split" / "train" / "images"

INT8_OUTPUT.mkdir(parents=True, exist_ok=True)

# Load calibration images (train split only)
print(f"Loading calibration images from: {CALIBRATION_IMAGES}")
image_files = sorted([
    f for f in os.listdir(CALIBRATION_IMAGES)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
print(f"Found {len(image_files)} calibration images")

# Use subset for faster calibration (~150 images is enough for NNCF)
# NNCF recommends 100-300 representative samples
calib_count = min(150, len(image_files))
# Use evenly spaced samples
indices = np.linspace(0, len(image_files) - 1, calib_count, dtype=int)
calib_files = [image_files[i] for i in indices]
print(f"Using {len(calib_files)} images for calibration")

def preprocess_image(img_path, target_size=(640, 640)):
    """Preprocess image for YOLOv8 OpenVINO inference.

    Matches ultralytics preprocessing: BGR→RGB, resize (letterbox), normalize to [0,1],
    HWC→CHW, add batch dim.

    Takes full image path as input (nncf.Dataset data_source item).
    Returns preprocessed numpy array in (1, 3, 640, 640) NCHW format.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return np.zeros((1, 3, 640, 640), dtype=np.float32)

    # Ultralytics preprocessing pipeline
    h0, w0 = img.shape[:2]
    # Calculate scale to fit within target_size
    r = min(target_size[0] / h0, target_size[1] / w0)
    new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
    # Resize
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Pad to target_size
    dh = target_size[0] - new_h
    dw = target_size[1] - new_w
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR → RGB
    img = img[..., ::-1]
    # HWC → CHW, normalize
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)  # => (1, 3, 640, 640)
    return np.ascontiguousarray(img)

# Create NNCF calibration dataset
# NNCF Dataset(data_source, transform_func):
#   data_source: iterable of raw data items
#   transform_func: takes one data_source item, returns model input
calib_paths = [str(CALIBRATION_IMAGES / f) for f in calib_files]
print(f"\nCreating calibration dataset with {len(calib_paths)} images...")
calib_dataset = nncf.Dataset(calib_paths, preprocess_image)

# Load FP32 model
print(f"\nLoading FP32 model from: {FP32_MODEL}")
core = ov.Core()
model = core.read_model(str(FP32_MODEL))

# Run NNCF quantization (Post-Training Quantization)
print("\nRunning NNCF INT8 quantization...")
print("This may take a few minutes...")
try:
    quantized_model = nncf.quantize(
        model,
        calib_dataset,
        preset=nncf.QuantizationPreset.MIXED,  # MIXED for better accuracy
        target_device=nncf.TargetDevice.CPU,
        subset_size=len(calib_files),
        fast_bias_correction=True,
    )
    print("Quantization succeeded!")
except Exception as e:
    print(f"NNCF quantize() failed: {e}")
    print("Trying alternative approach with compress_weights...")
    # Fallback: try with fewer samples
    quantized_model = nncf.quantize(
        model,
        calib_dataset,
        preset=nncf.QuantizationPreset.PERFORMANCE,
        target_device=nncf.TargetDevice.CPU,
        subset_size=min(50, len(calib_files)),
        fast_bias_correction=True,
    )

# Save INT8 model
output_xml = INT8_OUTPUT / "best_int8.xml"
output_bin = INT8_OUTPUT / "best_int8.bin"
print(f"\nSaving INT8 model to: {output_xml}")
ov.save_model(quantized_model, str(output_xml), compress_to_fp16=False)
print(f"INT8 model saved!")

# Print sizes
fp32_size = sum(
    f.stat().st_size
    for f in FP32_MODEL.parent.glob("*")
    if f.is_file()
)
int8_size = sum(
    f.stat().st_size
    for f in INT8_OUTPUT.glob("*")
    if f.is_file()
)
print(f"\n=== Model Sizes ===")
print(f"FP32 model: {fp32_size / 1024 / 1024:.2f} MB")
print(f"INT8 model: {int8_size / 1024 / 1024:.2f} MB")
print(f"Size reduction: {(1 - int8_size / fp32_size) * 100:.1f}%")
