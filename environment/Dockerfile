# hash:sha256:88bf4daba42ded1022bdbbf4e474c459239e656c72ec503607400557986991d6
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyterlab==3.1.4 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    altair==4.1.0 \
    matplotlib==3.4.2 \
    scipy==1.7.1
