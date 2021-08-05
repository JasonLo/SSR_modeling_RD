# hash:sha256:02aabadb5a208414836af8e1b98f864ac87787dc521bc7a2292fc80c919573b5
FROM registry.codeocean.com/codeocean/miniconda3:4.8.2-python3.8-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys \
        0xAD2323F17326AE31401037733E05EBFF05441C52 \
        E298A3A825C0D65DFD57CBB651716619E084DAB9 \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y 'deb http://deb.codeocean.com/rstudio-server-bionic/ ubuntu main' \
    && add-apt-repository -y 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/' \
    && apt-get purge -y --autoremove software-properties-common \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.4ubuntu1 \
        libgit2-dev=0.26.0+dfsg.1-1.1ubuntu0.2 \
        libssl-dev=1.1.1-1ubuntu2.1~18.04.9 \
        pandoc=1.19.2.4~dfsg-1build4 \
        pkg-config=0.29.1-0ubuntu2 \
        r-base=4.1.0-1.1804.0 \
        rstudio-server=1.2.5033 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyterlab==3.0.14 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    altair==4.1.0 \
    matplotlib \
    scipy

RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org/"), download.file.method = "libcurl")' >> $(Rscript -e 'cat(R.home())')/etc/Rprofile.site \
    && echo 'options(Ncpus = parallel::detectCores())' >> $(Rscript -e 'cat(R.home())')/etc/Rprofile.site \
    && Rscript -e 'options(warn=2); install.packages("remotes")'
RUN Rscript -e 'remotes::install_version("effectsize", "0.4.5")' \
    && Rscript -e 'remotes::install_version("insight", "0.14.2")' \
    && Rscript -e 'remotes::install_version("lme4")' \
    && Rscript -e 'remotes::install_version("lmerTest")' \
    && Rscript -e 'remotes::install_version("rlang", "0.4.11")' \
    && Rscript -e 'remotes::install_version("sjPlot")'
