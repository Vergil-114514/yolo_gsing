"""
N100 部署推理脚本 — OpenVINO INT8
===============================
用法:
  python inference_n100.py                          # 默认用摄像头
  python inference_n100.py --source ./test.jpg       # 单张图片
  python inference_n100.py --source ./images/        # 图片文件夹
  python inference_n100.py --source ./video.mp4      # 视频文件

依赖 (Ubuntu 22.04):
  pip install openvino opencv-python numpy
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


# ============================================================
# 配置
# ============================================================
CLASS_NAMES = ["Cube_food", "Cube_ins", "Cube_medicine", "Cube_tool"]
COLORS = [
    (0, 255, 0),     # Cube_food: 绿色
    (255, 0, 0),     # Cube_ins: 蓝色
    (0, 0, 255),     # Cube_medicine: 红色
    (255, 255, 0),   # Cube_tool: 青色
]
IMGSZ = 640
CFG = {"conf": 0.25, "iou": 0.45, "display": False}

# 默认模型路径（假设脚本和模型在同一目录结构下）
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_XML = SCRIPT_DIR / "runs" / "cube_yolov8n_n100" / "openvino_int8" / "best_int8.xml"


# ============================================================
# 预处理
# ============================================================
def letterbox(img, new_shape=640):
    """等比缩放 + 填充，匹配 YOLO 训练时预处理。"""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw = new_shape - new_w
    dh = new_shape - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (r, left, top)


def preprocess(img):
    """BGR 图像 → (1, 3, 640, 640) float32 NCHW."""
    img_padded, pad_info = letterbox(img, IMGSZ)
    img_rgb = img_padded[..., ::-1]                     # BGR → RGB
    img_norm = img_rgb.astype(np.float32) / 255.0       # [0, 1]
    img_chw = np.transpose(img_norm, (2, 0, 1))         # HWC → CHW
    img_batch = np.expand_dims(img_chw, axis=0)         # add batch
    return np.ascontiguousarray(img_batch), pad_info


# ============================================================
# 后处理
# ============================================================
def xywh2xyxy(boxes):
    """cxcywh → xyxy."""
    out = np.copy(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def nms(boxes, scores, iou_thres):
    """非极大值抑制。"""
    if len(boxes) == 0:
        return []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
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
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def postprocess(output, pad_info, orig_shape, conf_thres=0.25, iou_thres=0.45):
    """
    解析 YOLOv8 OpenVINO 输出 → 检测框列表。
    output: (1, 8, 8400)，通道 = [cx, cy, w, h, cls0, cls1, cls2, cls3]
    """
    output = np.squeeze(output).T   # (8400, 8)
    boxes_raw = output[:, :4]        # cx, cy, w, h
    scores = output[:, 4:]           # cls0..cls3

    max_scores = scores.max(axis=1)
    class_ids = scores.argmax(axis=1)

    mask = max_scores > conf_thres
    if not mask.any():
        return np.empty((0, 6))

    boxes_raw, max_scores, class_ids = boxes_raw[mask], max_scores[mask], class_ids[mask]
    boxes = xywh2xyxy(boxes_raw)

    # 从模型输入空间缩回原图
    r, left, top = pad_info
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top
    boxes[:, [0, 2]] /= r
    boxes[:, [1, 3]] /= r

    h0, w0 = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

    # 按类别做 NMS
    results = []
    for cls_id in range(len(CLASS_NAMES)):
        idx = np.where(class_ids == cls_id)[0]
        keep = nms(boxes[idx], max_scores[idx], iou_thres)
        for k in keep:
            results.append(np.concatenate([boxes[idx][k], [max_scores[idx][k], cls_id]]))
    if not results:
        return np.empty((0, 6))
    return np.array(results)


# ============================================================
# 绘图
# ============================================================
def draw_detections(img, detections, fps):
    """画框、标签、FPS。"""
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_id = int(cls_id)
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # FPS 左上角
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return img


# ============================================================
# 模型加载
# ============================================================
def load_model(xml_path):
    """加载 OpenVINO 模型，InferRequest 挂到 compiled 对象上复用。"""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"模型文件不存在: {xml_path}")
    core = ov.Core()
    print(f"[INFO] OpenVINO available devices: {core.available_devices}")

    # N100 无独显，固定 CPU
    device = "CPU"
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(model, device)
    # 预创建 InferRequest 并挂载，避免每帧 create_infer_request() 开销
    compiled._request = compiled.create_infer_request()
    print(f"[INFO] 模型已加载: {xml_path} -> {device}")
    return compiled


# ============================================================
# 推理
# ============================================================
def infer(compiled_model, img_tensor):
    """单次推理，复用预创建的 InferRequest。"""
    compiled_model._request.set_input_tensor(ov.Tensor(img_tensor))
    compiled_model._request.infer()
    return compiled_model._request.get_output_tensor().data


# ============================================================
# 主流程
# ============================================================
def process_image(compiled_model, img_path, output_dir=None):
    """单张图片推理。"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] 无法读取: {img_path}")
        return

    t0 = time.perf_counter()
    tensor, pad_info = preprocess(img)
    t1 = time.perf_counter()
    output = infer(compiled_model, tensor)
    t2 = time.perf_counter()
    detections = postprocess(output, pad_info, img.shape, CFG["conf"], CFG["iou"])
    t3 = time.perf_counter()

    pre_ms = (t1 - t0) * 1000
    inf_ms = (t2 - t1) * 1000
    post_ms = (t3 - t2) * 1000
    total_ms = (t3 - t0) * 1000
    fps = 1000 / total_ms

    print(f"[单张] 预处理={pre_ms:.1f}ms  推理={inf_ms:.1f}ms  后处理={post_ms:.1f}ms  总计={total_ms:.1f}ms  FPS={fps:.1f}")

    img_result = draw_detections(img.copy(), detections, fps)
    print(f"[检测] 共 {len(detections)} 个目标:")
    for det in detections:
        _, _, _, _, conf, cls_id = det
        print(f"  {CLASS_NAMES[int(cls_id)]}: {conf:.3f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, os.path.basename(img_path))
    else:
        out_path = os.path.splitext(img_path)[0] + "_detected.jpg"
    cv2.imwrite(out_path, img_result)
    print(f"[保存] {out_path}")


def process_folder(compiled_model, folder, output_dir=None):
    """批量图片推理。不指定 --output 时结果存到 folder_detected/，不会覆盖原图。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts)
    if not files:
        print(f"[ERROR] 目录下无图片: {folder}")
        return

    if output_dir is None:
        output_dir = folder.rstrip("/\\") + "_detected"
    os.makedirs(output_dir, exist_ok=True)

    total_times = []
    total_detections = 0
    for f in files:
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        t0 = time.perf_counter()
        tensor, pad_info = preprocess(img)
        output = infer(compiled_model, tensor)
        detections = postprocess(output, pad_info, img.shape, CFG["conf"], CFG["iou"])
        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000
        total_times.append(ms)
        total_detections += len(detections)

        fps = 1000 / ms
        print(f"[{f}] {ms:.1f}ms  FPS={fps:.1f}  {len(detections)} targets")

        img_result = draw_detections(img.copy(), detections, fps)
        out_path = os.path.join(output_dir, f)
        cv2.imwrite(out_path, img_result)

    avg_ms = np.mean(total_times)
    print(f"\n[汇总] {len(total_times)} 张图片, 平均 {avg_ms:.1f}ms, "
          f"平均 FPS={1000/avg_ms:.1f}, 共检测 {total_detections} 个目标")


def process_camera(compiled_model, camera_id=0):
    """摄像头实时推理。"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头 id={camera_id}")
        return

    if CFG["display"]:
        print("[实时] 按 'q' 退出")
    else:
        print("[实时] 无头模式，按 Ctrl+C 退出")
    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        tensor, pad_info = preprocess(frame)
        output = infer(compiled_model, tensor)
        detections = postprocess(output, pad_info, frame.shape, CFG["conf"], CFG["iou"])
        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000
        fps_history.append(1000 / ms)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        if CFG["display"]:
            frame_drawn = draw_detections(frame, detections, avg_fps)
            cv2.imshow("N100 Inference", frame_drawn)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if CFG["display"]:
        cv2.destroyAllWindows()
    print(f"[退出] 平均 FPS={np.mean(fps_history):.1f}")


def process_video(compiled_model, video_path, output_path=None):
    """视频文件推理。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps_src, (w, h))

    fps_history = []
    frame_count = 0
    print(f"[视频] {total_frames} 帧, 源帧率 {fps_src:.1f} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        tensor, pad_info = preprocess(frame)
        output = infer(compiled_model, tensor)
        detections = postprocess(output, pad_info, frame.shape, CFG["conf"], CFG["iou"])
        t1 = time.perf_counter()

        ms = (t1 - t0) * 1000
        fps_history.append(1000 / ms)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)
        frame_count += 1

        frame_drawn = draw_detections(frame, detections, avg_fps)
        if out:
            out.write(frame_drawn)

        if frame_count % 30 == 0:
            print(f"  进度 {frame_count}/{total_frames}  FPS={avg_fps:.1f}")

    if out:
        out.release()
    cap.release()
    if CFG["display"]:
        cv2.destroyAllWindows()
    print(f"[完成] {frame_count} 帧, 平均推理 FPS={np.mean(fps_history):.1f}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="N100 OpenVINO INT8 推理脚本")
    parser.add_argument("--source", "-s", type=str, default="camera",
                        help="输入源: 图片路径 / 文件夹 / 视频路径 / 'camera' (默认)")
    parser.add_argument("--model", "-m", type=str, default=str(DEFAULT_MODEL_XML),
                        help="OpenVINO INT8 模型 .xml 路径")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="摄像头编号 (默认 0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出目录 (图片/文件夹) 或输出视频路径 (视频)")
    parser.add_argument("--conf", type=float, default=CFG["conf"],
                        help=f"置信度阈值 (默认 {CFG['conf']})")
    parser.add_argument("--iou", type=float, default=CFG["iou"],
                        help=f"NMS IOU 阈值 (默认 {CFG['iou']})")
    parser.add_argument("--display", action="store_true",
                        help="显示画面窗口 (默认关闭，适合无头/SSH 环境)")
    args = parser.parse_args()

    CFG["conf"] = args.conf
    CFG["iou"] = args.iou
    CFG["display"] = args.display

    compiled = load_model(args.model)

    source = args.source
    if source == "camera":
        process_camera(compiled, args.camera_id)
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            process_video(compiled, source, args.output)
        else:
            process_image(compiled, source, args.output)
    elif os.path.isdir(source):
        process_folder(compiled, source, args.output)
    else:
        print(f"[ERROR] 无效输入源: {source}")


if __name__ == "__main__":
    main()
