FROM nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libboost-all-dev \
    libeigen3-dev \
    libceres-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    libopencv-dev \
    libfreeimage-dev \
    libsqlite3-dev \
    libflann-dev \
    libmetis-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libtbb-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://rclone.org/install.sh | bash

RUN git clone https://github.com/colmap/colmap.git /opt/colmap \
    && cmake -S /opt/colmap -B /opt/colmap/build -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_ENABLED=ON \
      -DGUI_ENABLED=OFF \
    && cmake --build /opt/colmap/build \
    && cmake --install /opt/colmap/build

RUN git clone https://github.com/nerfstudio-project/gsplat.git /opt/gsplat \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install -r /opt/gsplat/requirements.txt \
    && python3 -m pip install -e /opt/gsplat

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
