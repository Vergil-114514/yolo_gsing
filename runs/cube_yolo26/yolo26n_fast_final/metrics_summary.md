# YOLO26n Fast Final Metrics

- Dataset: `Cube.yolo26` (`train=832`, `valid=157`, `test=53`)
- Model: `yolo26n.pt`
- Training: 200 epochs, `imgsz=640`, `batch=16`, `patience=50`, `cos_lr=True`, `close_mosaic=15`
- Note: labels contain mixed boxes and polygons; Ultralytics trained/evaluated detection boxes only.

## Validation

- Best validation checkpoint: `weights/best.pt`
- Validation metrics: `P=0.953`, `R=0.983`, `mAP50=0.992`, `mAP50-95=0.875`

## Test Split

| Model | P | R | mAP50 | mAP50-95 | Measured inference |
|------|---:|---:|---:|---:|---:|
| PyTorch `best.pt` | 0.967 | 0.978 | 0.963 | 0.864 | 14.9 ms/image on RTX/CUDA validation path |
| OpenVINO FP32 batch=1 | 0.953 | 0.960 | 0.967 | 0.873 | 68.3 ms/image on AMD 7840H CPU |
| OpenVINO INT8 batch=1 | 0.950 | 0.978 | 0.973 | 0.867 | 56.5 ms/image on AMD 7840H CPU |

Primary deployment artifact for Intel N100 single-image inference:

`weights/best_int8_openvino_model/`
