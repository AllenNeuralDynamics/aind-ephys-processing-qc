# hash:sha256:80087995085b139fb824fc73c59af31b9fd9ca1f7fb1cdf9e7542d31741c6d4b
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
    s3fs==2025.5.1 \
    spikeinterface[full,widgets]==0.105.3 \
    wavpack-numcodecs==0.2.2
