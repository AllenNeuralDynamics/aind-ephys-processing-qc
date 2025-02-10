# hash:sha256:40c2a3b197387759534a12d55359fa1560a8dc639e2dff1c5161c50e01f61574
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

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
