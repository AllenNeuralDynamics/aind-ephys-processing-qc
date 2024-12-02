# hash:sha256:c0903db98465a1337fe7d5a65ebb3e80f7fbae6d694379eccc8c36adaf2b2767
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.1.1 \
    aind-qcportal-schema==0.1.1 \
    spikeinterface[full,widgets]==0.101.2 \
    wavpack-numcodecs==0.2.1

COPY postInstall /
RUN /postInstall
