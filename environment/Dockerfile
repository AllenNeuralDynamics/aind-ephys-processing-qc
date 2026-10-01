# hash:sha256:6cc6ba5afb3e41447f888df49205e5b0e844674ac7db1247c4c27bb39101a7a1
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
    spikeinterface[full,widgets]==0.105.0 \
    wavpack-numcodecs==0.2.2
