# hash:sha256:3370166b68dfdcf09909891ce882367ad687333e6715b246905adfabbd9e773e
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
