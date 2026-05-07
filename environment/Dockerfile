# hash:sha256:6127992336e36cc3b0b727396441f8ba4067168830172561f3a1b73c8e44ee43
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.4.0 \
    aind-log-utils==0.2.3 \
    aind-qcportal-schema==0.6.4 \
    s3fs==2025.5.1 \
    spikeinterface[full,widgets]==0.103.2 \
    wavpack-numcodecs==0.2.2
