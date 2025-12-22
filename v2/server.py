import cv2
import mediapipe as mp
import asyncio
import websockets
import json

# 配置
RTSP_URL = "rtsp://YOUR_IPHONE_IP:PORT/live"  # 替换为你iPhone App里的地址
PORT = 8765

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 在Orin上可以尝试改为 2 获得更高精度
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


async def handler(websocket):
    print("Client connected")
    cap = cv2.VideoCapture(RTSP_URL)

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # 转换为 RGB (MediaPipe 需要)
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 推理
            results = pose.process(image_rgb)

            # 提取数据
            data = {}
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append(
                        {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    )
                data["landmarks"] = landmarks
                data["found"] = True
            else:
                data["found"] = False

            # 发送 JSON 数据给前端
            await websocket.send(json.dumps(data))

            # 控制帧率，避免堵塞网络 (根据需要调整)
            await asyncio.sleep(0.01)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        cap.release()


async def main():
    print(f"Server started on ws://0.0.0.0:{PORT}")
    async with websockets.serve(handler, "0.0.0.0", PORT):
        await asyncio.future  # run forever


if __name__ == "__main__":
    asyncio.run(main())
