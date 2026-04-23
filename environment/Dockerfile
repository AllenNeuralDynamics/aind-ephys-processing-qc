# hash:sha256:d80ac4150af3c1d4d50de73e60c74124c62511ff525321bf51523a9f2467994b
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
    matplotlib-venn==1.1.2 \
    s3fs==2025.5.1 \
    spikeinterface[full,widgets]==0.104.3 \
    wavpack-numcodecs==0.2.2
