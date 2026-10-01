# hash:sha256:35216c27b355cfb5ead1a9fdf6f776f269193c93f7799ab6a3941b3df5d3e8a0
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
    aind-data-schema==2.8.1 \
    aind-qcportal-schema==0.6.4 \
    log-schema==0.2.10.dev1 \
    matplotlib-venn==1.1.2 \
    spikeinterface[full,widgets]==0.105.0 \
    wavpack-numcodecs==0.2.2
