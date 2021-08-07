# hash:sha256:c9fbba972efd54286aadb1811fc2e7080abe574cb136b497133570299ce6eacc
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyterlab==3.1.4 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    altair==4.1.0 \
    matplotlib==3.4.2 \
    scipy==1.7.1
