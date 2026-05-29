"""
Validate and benchmark OpenVINO FP32 vs INT8 models on the test set.
Uses OpenVINO directly for inference, computes mAP50 using same pipeline.
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import cv2
import time
import json
from collections import defaultdict

import openvino as ov
import openvino.runtime.op as real_op

# Monkey-patch for NNCF compatibility (needed if we re-import)
import sys as _sys
_sys.modules['openvino.op'] = real_op

BASE = Path(r"e:/xwechat_files/wxid_el79rox3ahjk12_09c9/msg/file/2026-05/yolo_gsing")
RUNS = BASE / "runs" / "cube_yolov8n_n100"
SPLIT = BASE / "Cube.yolov8" / "split"
TEST_IMAGES = SPLIT / "test" / "images"
TEST_LABELS = SPLIT / "test" / "labels"

# Model paths
FP32_XML = RUNS / "openvino_fp32" / "best.xml"
INT8_XML = RUNS / "openvino_int8" / "best_int8.xml"

# YOLO constants
NC = 4
NAMES = ['Cube_food', 'Cube_ins', 'Cube_medicine', 'Cube_tool']
IMGSZ = 640
CONF_THRES = 0.001  # Low conf for mAP, NMS handles rest
IOU_THRES = 0.7

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image to new_shape, preserving aspect ratio."""
    shape = img.shape[:2]  # current [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2  # divide padding into 2 sides
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, dw, dh, top, left)

def preprocess_image(img_path):
    """Preprocess image for YOLOv8 OpenVINO inference."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None
    h0, w0 = img.shape[:2]
    img_padded, (r, dw, dh, top, left) = letterbox(img, (IMGSZ, IMGSZ))
    img_padded = img_padded[..., ::-1]  # BGR to RGB
    img_padded = img_padded.astype(np.float32) / 255.0
    img_padded = np.transpose(img_padded, (2, 0, 1))  # HWC to CHW
    img_padded = np.expand_dims(img_padded, axis=0)    # (1, 3, 640, 640)
    return np.ascontiguousarray(img_padded), (h0, w0, r, dw, dh, top, left)

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def scale_boxes(boxes, preprocess_info, orig_shape):
    """Scale boxes from model input size back to original image size."""
    h0, w0, r, dw, dh, top, left = preprocess_info
    # Remove padding
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    # Scale
    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r
    # Clip
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
    return boxes

def nms(boxes, scores, iou_thres=0.7):
    """Simple NMS implementation."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

def load_model(xml_path, device="CPU"):
    """Load OpenVINO model."""
    core = ov.Core()
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(model, device)
    return compiled

def predict(compiled_model, img_tensor):
    """Run inference on a single image."""
    infer_request = compiled_model.create_infer_request()
    input_tensor = ov.Tensor(img_tensor)
    infer_request.set_input_tensor(input_tensor)
    infer_request.infer()
    output = infer_request.get_output_tensor().data
    return output  # (1, 8, 8400) - [cx, cy, w, h, obj, cls0, cls1, cls2, cls3]

def postprocess(output, preprocess_info, orig_shape, conf_thres=0.25, iou_thres=0.7):
    """Post-process YOLOv8 output to get detections."""
    output = np.squeeze(output)  # (8, 8400)
    output = output.T  # (8400, 8)

    # Split outputs
    boxes_raw = output[:, :4]  # cx, cy, w, h
    scores = output[:, 4:]  # obj, cls0, cls1, cls2, cls3

    # Get class scores
    class_scores = scores[:, 1:]  # cls0..cls3
    max_class_scores = class_scores.max(axis=1)
    max_class_ids = class_scores.argmax(axis=1)

    # Filter by confidence
    mask = max_class_scores > conf_thres
    boxes_raw = boxes_raw[mask]
    scores_filtered = max_class_scores[mask]
    class_ids = max_class_ids[mask]

    # Convert boxes from cxcywh to xyxy
    boxes = xywh2xyxy(boxes_raw)

    # Scale boxes back to original image size
    boxes = scale_boxes(boxes, preprocess_info, orig_shape[:2])

    # NMS per class
    final_boxes = []
    final_scores = []
    final_cls = []

    for cls_id in range(NC):
        cls_mask = class_ids == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores_filtered[cls_mask]

        keep = nms(cls_boxes, cls_scores, iou_thres)
        for idx in keep:
            final_boxes.append(cls_boxes[idx])
            final_scores.append(cls_scores[idx])
            final_cls.append(cls_id)

    if len(final_boxes) == 0:
        return np.zeros((0, 6))

    return np.column_stack([final_boxes, final_cls, final_scores])

def load_labels(label_path):
    """Load YOLO format labels."""
    labels = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    labels.append([cls_id, cx, cy, w, h])
    return np.array(labels) if labels else np.zeros((0, 5))

def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-16)

def compute_ap(recall, precision):
    """Compute AP using 11-point interpolation."""
    recall = np.array(recall)
    precision = np.array(precision)
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        if len(p) > 0:
            ap += np.max(p) / 11
    return ap

def evaluate(model, model_name, device="CPU"):
    """Evaluate model on test set, compute mAP50 and speed metrics."""
    compiled = load_model(model, device)

    # Warmup
    dummy = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    for _ in range(5):
        predict(compiled, dummy)

    # Gather all images
    image_files = sorted([
        f for f in os.listdir(TEST_IMAGES)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {len(image_files)} test images")
    print(f"{'='*60}")

    # For mAP computation
    all_detections = []  # List of (image_id, box, score, class)
    all_groundtruths = []  # List of (image_id, box, class)
    inference_times = []

    for img_idx, img_file in enumerate(image_files):
        img_path = TEST_IMAGES / img_file
        label_path = TEST_LABELS / (os.path.splitext(img_file)[0] + ".txt")

        # Preprocess
        img_tensor, preprocess_info = preprocess_image(img_path)
        if img_tensor is None:
            continue

        h0, w0 = img_tensor.shape[2:]
        # Actually read original image for shape
        orig_img = cv2.imread(str(img_path))
        orig_shape = orig_img.shape

        # Inference with timing
        t0 = time.perf_counter()
        output = predict(compiled, img_tensor)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)  # ms

        # Post-process
        detections = postprocess(output, preprocess_info, orig_shape, conf_thres=CONF_THRES, iou_thres=IOU_THRES)

        # Store detections
        for det in detections:
            all_detections.append((img_idx, det[:4], det[5], int(det[4])))

        # Load and store ground truths
        labels = load_labels(label_path)
        for label in labels:
            cls_id = int(label[0])
            cx, cy, w, h = label[1], label[2], label[3], label[4]
            # Convert normalized cxcywh to xyxy in pixel coords
            x1 = (cx - w/2) * orig_shape[1]
            y1 = (cy - h/2) * orig_shape[0]
            x2 = (cx + w/2) * orig_shape[1]
            y2 = (cy + h/2) * orig_shape[0]
            all_groundtruths.append((img_idx, [x1, y1, x2, y2], cls_id))

        if (img_idx + 1) % 10 == 0:
            print(f"  Processed {img_idx + 1}/{len(image_files)} images...")

    # --- Compute mAP ---
    # Group by class
    detections_by_class = defaultdict(list)
    for img_id, box, score, cls_id in all_detections:
        detections_by_class[cls_id].append((img_id, box, score))

    gt_by_class = defaultdict(lambda: defaultdict(list))
    for img_id, box, cls_id in all_groundtruths:
        gt_by_class[cls_id][img_id].append(box)

    # Count total GTs per class
    gt_counts = {}
    for cls_id in range(NC):
        gt_counts[cls_id] = sum(len(gt_by_class[cls_id][i]) for i in gt_by_class[cls_id])

    aps = {}
    precisions = {}
    recalls = {}

    for cls_id in range(NC):
        dets = sorted(detections_by_class[cls_id], key=lambda x: x[2], reverse=True)
        npos = gt_counts[cls_id]

        if npos == 0:
            aps[cls_id] = 0.0
            precisions[cls_id] = 0.0
            recalls[cls_id] = 0.0
            continue

        # Handle case with no detections but has ground truths
        if len(dets) == 0:
            aps[cls_id] = 0.0
            precisions[cls_id] = 0.0
            recalls[cls_id] = 0.0
            continue

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        gt_matched = defaultdict(set)

        for i, (img_id, box, score) in enumerate(dets):
            if img_id in gt_by_class[cls_id]:
                gts = gt_by_class[cls_id][img_id]
                max_iou = 0
                max_j = -1
                for j, gt_box in enumerate(gts):
                    iou = compute_iou(box, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_j = j
                if max_iou >= 0.5 and max_j not in gt_matched[img_id]:
                    tp[i] = 1
                    gt_matched[img_id].add(max_j)
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        rec = tp_cumsum / npos
        prec = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)

        aps[cls_id] = compute_ap(rec, prec)

        # Get precision and recall at the best F1 point (or at recall ~0.5)
        # For simplicity, report average over all thresholds
        precisions[cls_id] = np.mean(prec) if len(prec) > 0 else 0
        recalls[cls_id] = np.mean(rec) if len(rec) > 0 else 0

    mAP50 = np.mean(list(aps.values()))

    # --- Speed metrics ---
    times = np.array(inference_times)
    avg_latency = np.mean(times)
    p95_latency = np.percentile(times, 95)
    fps = 1000.0 / avg_latency if avg_latency > 0 else float('inf')

    # --- Print results ---
    print(f"\n--- {model_name} Results ---")
    print(f"{'Class':<20} {'AP50':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 52)
    for cls_id in range(NC):
        print(f"{NAMES[cls_id]:<20} {aps[cls_id]:<10.4f} {precisions[cls_id]:<12.4f} {recalls[cls_id]:<10.4f}")
    print("-" * 52)
    print(f"{'mAP50':<20} {mAP50:<10.4f}")
    print(f"\nSpeed (CPU: {device}):")
    print(f"  Average latency: {avg_latency:.2f} ms")
    print(f"  P95 latency:     {p95_latency:.2f} ms")
    print(f"  FPS:             {fps:.2f}")

    # --- Model size ---
    model_size = sum(f.stat().st_size for f in Path(model).parent.glob("*") if f.is_file())
    print(f"  Model size:      {model_size / 1024 / 1024:.2f} MB")

    results = {
        "model": model_name,
        "mAP50": float(mAP50),
        "per_class_ap50": {NAMES[i]: float(aps[i]) for i in range(NC)},
        "precision": {NAMES[i]: float(precisions[i]) for i in range(NC)},
        "recall": {NAMES[i]: float(recalls[i]) for i in range(NC)},
        "avg_latency_ms": float(avg_latency),
        "p95_latency_ms": float(p95_latency),
        "fps": float(fps),
        "model_size_mb": model_size / 1024 / 1024,
    }
    return results

# --- Main ---
if __name__ == "__main__":
    print("Loading models...")

    # Evaluate FP32
    fp32_results = evaluate(FP32_XML, "OpenVINO FP32", device="CPU")

    # Evaluate INT8
    int8_results = evaluate(INT8_XML, "OpenVINO INT8", device="CPU")

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'FP32':<15} {'INT8':<15} {'Delta':<15}")
    print("-" * 70)

    mAP50_delta_pct = ((int8_results['mAP50'] - fp32_results['mAP50']) / fp32_results['mAP50']) * 100
    fps_delta_pct = ((int8_results['fps'] - fp32_results['fps']) / fp32_results['fps']) * 100
    size_delta_pct = ((int8_results['model_size_mb'] - fp32_results['model_size_mb']) / fp32_results['model_size_mb']) * 100

    print(f"{'mAP50':<25} {fp32_results['mAP50']:<15.4f} {int8_results['mAP50']:<15.4f} {mAP50_delta_pct:+.2f}%")
    print(f"{'Avg Latency (ms)':<25} {fp32_results['avg_latency_ms']:<15.2f} {int8_results['avg_latency_ms']:<15.2f} {(int8_results['avg_latency_ms'] - fp32_results['avg_latency_ms']):+.2f} ms")
    print(f"{'P95 Latency (ms)':<25} {fp32_results['p95_latency_ms']:<15.2f} {int8_results['p95_latency_ms']:<15.2f} {(int8_results['p95_latency_ms'] - fp32_results['p95_latency_ms']):+.2f} ms")
    print(f"{'FPS':<25} {fp32_results['fps']:<15.2f} {int8_results['fps']:<15.2f} {fps_delta_pct:+.2f}%")
    print(f"{'Model Size (MB)':<25} {fp32_results['model_size_mb']:<15.2f} {int8_results['model_size_mb']:<15.2f} {size_delta_pct:+.2f}%")

    # --- Pass/Fail ---
    print(f"\n{'='*60}")
    print("ACCEPTANCE CRITERIA")
    print(f"{'='*60}")

    mAP_pass = abs(mAP50_delta_pct) <= 5.0
    fps_pass = fps_delta_pct >= 50.0

    print(f"  mAP50 relative loss <= 5%:  {'PASS' if mAP_pass else 'FAIL'} ({abs(mAP50_delta_pct):.2f}%)")
    print(f"  FPS improvement >= 50%:     {'PASS' if fps_pass else 'FAIL'} ({fps_delta_pct:+.2f}%)")

    # Save results
    import csv
    csv_path = RUNS / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "mAP50", "Avg_Latency_ms", "P95_Latency_ms", "FPS", "Model_Size_MB"])
        for results in [fp32_results, int8_results]:
            writer.writerow([
                results["model"],
                f"{results['mAP50']:.4f}",
                f"{results['avg_latency_ms']:.2f}",
                f"{results['p95_latency_ms']:.2f}",
                f"{results['fps']:.2f}",
                f"{results['model_size_mb']:.2f}",
            ])
    print(f"\nBenchmark results saved to: {csv_path}")

    # Save JSON
    json_path = RUNS / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump({"fp32": fp32_results, "int8": int8_results}, f, indent=2)
    print(f"Detailed results saved to: {json_path}")
