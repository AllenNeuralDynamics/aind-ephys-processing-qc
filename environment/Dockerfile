# hash:sha256:43ca989113ae96cee8c80942078c26e79d57b35e0215506649fec49d0b7df3cc
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
    aind-data-schema==2.0.1 \
    aind-log-utils==0.2.3 \
    aind-qcportal-schema==0.4.0 \
    s3fs==2025.5.1 \
    spikeinterface[full,widgets]==0.103.0 \
    wavpack-numcodecs==0.2.2
