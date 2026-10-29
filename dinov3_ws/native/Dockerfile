ARG BASE_IMAGE=advigw/isaac-ros-base-jazzy:1.00
FROM ${BASE_IMAGE}
USER root
COPY nvdcf_bridge.cpp build_nvdcf.sh install_nvdcf.sh /tmp/nvdcf/native/
RUN bash /tmp/nvdcf/native/install_nvdcf.sh \
    && install -m 0755 /tmp/nvdcf/artifacts/nvdcf/nvdcf_bridge /usr/local/bin/dinov3-nvdcf-bridge \
    && rm -rf /tmp/nvdcf /tmp/deepstream-9.1-arm64.deb /var/lib/apt/lists/*
