# WeDance: Jetson Orin Nano Pose Estimation Deployment Guide

This project establishes a real-time human pose estimation system using the **NVIDIA Jetson Orin Nano** and the **Ultralytics YOLOv11-Pose** model. The system is containerized using Docker, leverages TensorRT for accelerated inference, and processes RTSP video streams from iOS devices.

## 📋 Table of Contents

1. [Environment Setup (Docker)](https://www.google.com/search?q=%231-environment-setup-docker)
2. [RTSP Stream Testing](https://www.google.com/search?q=%232-rtsp-stream-testing)
3. [Model Preparation (TensorRT)](https://www.google.com/search?q=%233-model-preparation-tensorrt)
4. [Service Deployment](https://www.google.com/search?q=%234-service-deployment)

---

## 1. Environment Setup (Docker)

We use the official Ultralytics Docker image optimized for JetPack 6.

Reference: [NVIDIA Jetson Guide](https://docs.ultralytics.com/guides/nvidia-jetson/)

### Pull and Test the Image

Ensure `nvidia-container-runtime` is installed on your Jetson.

```bash
# Define the image tag (for JetPack 6)
export t=ultralytics/ultralytics:latest-jetson-jetpack6

# Pull the image
sudo docker pull $t

# Run an interactive container for testing (auto-removes on exit)
# Note: --ipc=host is used to share memory segments and prevent crashes
sudo docker run -it --ipc=host --runtime=nvidia $t

```

---

## 2. RTSP Stream Testing

Before deploying the service, verify that the Jetson can receive the video stream from your iOS device.

* **Source**: iOS Device (using an IP Camera App)
* **Receiver**: Jetson Orin Nano

```bash
# Use ffplay to test connectivity and latency
# -rtsp_transport tcp: Force TCP to prevent artifacts/packet loss
ffplay -rtsp_transport tcp rtsp://10.0.0.75:8554/stream

```

> **Note**: Replace the IP address (`10.0.0.75`) and path (`/stream`) with your actual values.

---

## 3. Model Preparation (TensorRT)

To achieve optimal performance on the Jetson, export the PyTorch model (`.pt`) to a TensorRT engine (`.engine`).

Reference: [Ultralytics Pose Docs](https://docs.ultralytics.com/tasks/pose/#models)

### Export Steps

1. Ensure you are inside the Docker container or have the `ultralytics` environment set up.
2. Run the following Python script to export the `yolo11n-pose` model:

```python
from ultralytics import YOLO

# Load the model
model = YOLO('yolo11n-pose.pt')

# Export to TensorRT format
# device=0: Use GPU
# half=True: Enable FP16 half-precision (Recommended for Orin Nano)
model.export(format='engine', device=0, half=True)

```

Once completed, a `yolo11n-pose.engine` file will be generated.

---

## 4. Service Deployment

### 4.1 Port Check

Before deploying, check if port `8765` is already in use.

```bash
# Method 1: Using lsof
sudo lsof -i :8765

# Method 2: Using netstat
sudo netstat -tulnp | grep 8765

```

If the port is occupied, stop the conflicting process or change the port in your configuration.

### 4.2 Start the Container

Start the production container. We will mount the host project directory into the container and map the WebSocket port.

* **Host Directory**: `/home/jetson/WeDance`
* **Container Directory**: `/home/jetson/WeDance`
* **Port Mapping**: `8765:8765`

```bash
# Define image variable
export t=ultralytics/ultralytics:latest-jetson-jetpack6

# Run container (Detached mode)
sudo docker run -dit \
    --runtime nvidia \
    --ipc=host \
    -p 8765:8765 \
    -v /home/jetson/WeDance:/home/jetson/WeDance \
    --name wedance001 \
    $t

```

### 4.3 Run Inference Service

Enter the container and start the Python server.

```bash
# Enter the container
sudo docker exec -it wedance001 /bin/bash

# (Inside Container) Navigate to project dir and run
cd /home/jetson/WeDance
python3 server.py

```

