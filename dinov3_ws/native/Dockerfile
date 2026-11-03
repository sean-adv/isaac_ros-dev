ARG BASE_IMAGE=advigw/isaac-ros-base-jazzy:1.00
FROM ${BASE_IMAGE}
USER root
# Keep dependency installation cached when native sources change.
COPY install_deps.sh /tmp/install_deps.sh
RUN bash /tmp/install_deps.sh \
    && rm -f /tmp/install_deps.sh /tmp/deepstream-9.1-arm64.deb \
    && rm -rf /var/lib/apt/lists/*
COPY . /tmp/nvdcf/native/
RUN bash /tmp/nvdcf/native/build.sh \
    && cmake --install /tmp/nvdcf/build/native --prefix /usr/local \
    && ldconfig \
    && rm -rf /tmp/nvdcf
