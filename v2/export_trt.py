from ultralytics import YOLO

# 1. 下载 PyTorch 模型 (.pt)
model = YOLO("yolov8n-pose.pt")

# 2. 导出为 TensorRT 引擎 (.engine)
# device=0 使用 GPU
# half=True 开启半精度推理 (FP16)，在 Orin 上能翻倍速度且不损失精度
model.export(format="engine", device=0, half=True)
