ARG BASE_IMAGE=advigw/isaac-ros-base-jazzy:1.00
FROM ${BASE_IMAGE}
USER root
COPY nvdcf_bridge.cpp build_nvdcf.sh install_nvdcf.sh gpu_geometry.cu build_gpu_geometry.sh video_gpu_bridge.cpp build_video_gpu.sh /tmp/nvdcf/native/
RUN bash /tmp/nvdcf/native/install_nvdcf.sh \
    && bash /tmp/nvdcf/native/build_video_gpu.sh \
    && install -m 0755 /tmp/nvdcf/artifacts/cuda/libgpu_geometry.so /usr/local/lib/libgpu_geometry.so \
    && install -m 0755 /tmp/nvdcf/artifacts/video/video_gpu_bridge /usr/local/bin/dinov3-video-gpu-bridge \
    && ldconfig \
    && install -m 0755 /tmp/nvdcf/artifacts/nvdcf/nvdcf_bridge /usr/local/bin/dinov3-nvdcf-bridge \
    && rm -rf /tmp/nvdcf /tmp/deepstream-9.1-arm64.deb /var/lib/apt/lists/*
