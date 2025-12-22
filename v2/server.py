import os
import cv2
import asyncio
import websockets
import json
from ultralytics import YOLO

# ================= 配置 =================
RTSP_URL = "rtsp://10.0.0.75:8554/stream"  # 记得修改 IP
PORT = 8765
MODEL_PATH = "yolo11n-pose.engine"  # 使用刚才生成的 engine 文件
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 加载 TensorRT 模型 (Task='pose')
print(f"Loading TensorRT Engine: {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH, task="pose")
except Exception as e:
    print(f"模型加载失败，请检查路径: {e}")
    exit()


async def handler(websocket):
    print(f"Client connected from {websocket.remote_address}")

    # 使用 GStreamer 管道进行硬解码 (性能优化版)
    # 如果这行报错，请回退到 cap = cv2.VideoCapture(RTSP_URL)
    gst_pipeline = (
        f"rtspsrc location={RTSP_URL} latency=200 ! "
        "qt-demux ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )
    # cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    # 或者简单版
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # === 调试点 1 ===
    print("摄像头初始化完成，准备进入循环...")
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                # === 调试点 2：高概率这里疯狂输出 ===
                print("❌ 读取帧失败 (Empty Frame)")
                await asyncio.sleep(0.1)
                continue

            # === YOLO 推理 ===
            # stream=True 减少内存占用
            results = model(frame, stream=True, verbose=False)

            data = {"found": False}

            for result in results:
                # 检查是否检测到人
                if result.keypoints is not None and len(result.keypoints) > 0:
                    # 获取归一化坐标 (0-1)
                    # shape: (1, 17, 2)
                    all_kpts = result.keypoints.xyn.cpu().numpy()
                    all_confs = result.keypoints.conf.cpu().numpy()

                    # 只要置信度最高的那个人
                    if len(all_kpts) > 0:
                        kpts = all_kpts[0]
                        confs = all_confs[0]
                        landmarks = []
                        for i, (x, y) in enumerate(kpts):
                            # YOLO 是 2D，没有 Z，设为 0
                            # 可见度使用置信度代替
                            visibility = float(confs[i])

                            # 只有置信度 > 0 且坐标不为 0 时才视为有效
                            if x == 0 and y == 0:
                                visibility = 0.0

                            landmarks.append(
                                {
                                    "x": float(x),
                                    "y": float(y),
                                    "z": 0.0,
                                    "visibility": visibility,
                                }
                            )

                        data["landmarks"] = landmarks
                        data["found"] = True
                        break  # 只发一个人
            # === 调试点 3 ===
            if data["found"]:
                print("✅ 找到人了！正在发送数据...")
            else:
                print("👀 画面正常，但没人")
            await websocket.send(json.dumps(data))
            # YOLO 极快，如果不加 sleep，可能会瞬间发太多包淹没前端
            # 但为了流畅度，我们不加 sleep，全速运行
            # 让出控制权，保持 WebSocket 心跳
            await asyncio.sleep(0)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"发生未知错误: {e}")
        # 打印错误但不退出程序，防止偶尔的坏帧搞挂服务
        import traceback

        traceback.print_exc()
    finally:
        cap.release()
        print("资源已释放")


async def main():
    print(f"Server started on ws://0.0.0.0:{PORT}")
    # 设置 ping_interval=None 可以防止某些因网络卡顿导致的意外断连
    async with websockets.serve(handler, "0.0.0.0", PORT, ping_interval=None):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
