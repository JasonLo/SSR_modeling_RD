# hash:sha256:7312cb74f2fdd70172a6436b3e416139e512d25432f52e132a986add2a3ba822
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyterlab==3.0.14 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    altair==4.1.0
