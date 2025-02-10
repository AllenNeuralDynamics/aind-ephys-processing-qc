# hash:sha256:3c5d0093c8aa045e629802002af7c467fa00fa45a3aa0a10ee0661d5aa7a3281
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.1.1 \
    aind-log-utils==0.2.3 \
    aind-qcportal-schema==0.4.0 \
    spikeinterface[full,widgets]==0.102.0 \
    wavpack-numcodecs==0.2.2
