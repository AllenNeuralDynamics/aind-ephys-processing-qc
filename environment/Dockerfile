# hash:sha256:56b357a534f8f821fc9537471d27003234ac8b0b644986bb8020229b722e9810
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.1.1 \
    aind-qcportal-schema==0.1.1 \
    matplotlib==3.9.2 \
    spikeinterface[full]==0.101.2
