FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Remove NVIDIA apt sources from the base image.
RUN find /etc/apt/ -type f -name "*.list" -exec sed -i '/repo.download.nvidia.com/d' {} +

RUN apt-get update && apt-get install -y \
    python3-pip \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    libjpeg-dev \
    python3-setuptools \
    python3-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Preserve the Jetson OpenCV installation.
RUN pip3 install --upgrade pip && \
    git clone --branch v8.2.100 https://github.com/ultralytics/ultralytics.git /tmp/ultralytics && \
    cd /tmp/ultralytics && \
    sed -i '/opencv-python/d' pyproject.toml && \
    pip3 install --no-cache-dir . && \
    rm -rf /tmp/ultralytics

RUN pip3 install --no-cache-dir pygame

COPY . .

CMD ["python3", "main.py"]
