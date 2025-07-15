# hash:sha256:78f1f550e6e3e62d7aef2f3a8b887967fdc196f8bc6d25bc1ea9554da1598ac8
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.3.0 \
    aind-log-utils==0.2.3 \
    aind-qcportal-schema==0.4.0 \
    s3fs==2025.5.1 \
    spikeinterface[full,widgets]==0.102.3 \
    wavpack-numcodecs==0.2.2

COPY postInstall /
RUN /postInstall
