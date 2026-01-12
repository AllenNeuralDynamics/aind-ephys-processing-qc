# hash:sha256:3c364a884735ee73aa9a180ff6a3c4388b0b0d30dd8a86922d3e6a0a022474c4
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.4.0 \
    aind-log-utils==0.2.3 \
    aind-qcportal-schema==0.6.4 \
    s3fs==2025.5.1 \
    spikeinterface[full,widgets]==0.103.0 \
    wavpack-numcodecs==0.2.2
