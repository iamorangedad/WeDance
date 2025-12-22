import cv2
import time

# 视频路径
video_path = "test_video.mp4"

# 构建 GStreamer 管道
# 1. filesrc: 读文件
# 2. nvv4l2decoder: 调用 NVDEC 硬件
# 3. nvvidconv: 把 NVDEC 的输出格式转为 OpenCV 能懂的 BGR
pipeline = (
    f"filesrc location={video_path} ! "
    "qtdemux ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

print(f"启动管道: {pipeline}")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ 无法打开视频，请检查路径或 GStreamer 插件")
    exit()

fps_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fps_count += 1

    # 为了测试纯解码性能，我们不显示画面 (cv2.imshow)，只空转
    if fps_count % 30 == 0:
        print(f"已解码 {fps_count} 帧", end="\r")

end_time = time.time()
avg_fps = fps_count / (end_time - start_time)

print(f"\n✅ 测试结束。平均帧率: {avg_fps:.2f} FPS")
cap.release()
