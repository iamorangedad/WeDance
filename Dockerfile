# 使用包含完整开发工具链的 JetPack 镜像 (包含 CUDA, cuDNN, TensorRT 等)
# 确保你的宿主机刷的是 JetPack 6.x 版本
FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0

# 避免交互报错
ENV DEBIAN_FRONTEND=noninteractive
ENV OPENCV_VERSION=4.8.0

# 1. 安装基础编译工具和 GStreamer 开发库
# 注意：Jetson 上编译不需要安装 cuda/cudnn 的 apt 包，因为基础镜像里已经有了
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-numpy \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. 下载源码
WORKDIR /tmp
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip opencv_contrib.zip

# 3. 编译 OpenCV (针对 Orin Nano 优化)
WORKDIR /tmp/opencv-${OPENCV_VERSION}/build

# 关键参数解析：
# -D CUDA_ARCH_BIN=8.7  -> 专门针对 Orin 架构
# -D WITH_CUBLAS=1      -> 必须开启，否则部分 CUDA 功能报错
# -D ENABLE_FAST_MATH=1 -> 提升性能
RUN cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D WITH_CUBLAS=1 \
    -D CUDA_ARCH_BIN=8.7 \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_GSTREAMER=ON \
    -D WITH_LIBV4L=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    # 这一行用于修复 Jetson 上 OpenGL 的路径问题 (可选，视具体情况)
    -D WITH_OPENGL=ON \
    .. && \
    # ⚠️ 警告：Orin Nano 上不要用 -j$(nproc)，内存会爆！建议 -j4
    make -j4 && \
    make install && \
    # 清理源码以减小体积
    rm -rf /tmp/opencv*

# 4. 配置动态链接库
RUN ldconfig

# 5. 验证命令
CMD ["python3", "-c", "import cv2; print(f'OpenCV: {cv2.__version__}'); info=cv2.getBuildInformation(); print('GStreamer:', 'YES' if 'GStreamer: YES' in info else 'NO'); print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"]