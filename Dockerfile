FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# install python
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo


RUN apt update && \
    apt install -y git && \
    apt install -y wget && \
    apt install -y unzip

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.8 python3-pip python3-setuptools python3-distutils python3-tk

# default workdir
WORKDIR /app

RUN pip install --upgrade pip
# create a symbolic link for python3.8 as python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# install jax
RUN pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# user: install jaxmarl
# RUN pip install jaxmarl

# dev: install from source
RUN git clone https://github.com/FLAIROx/JaxMARL.git && cd JaxMARL && pip install --no-cache-dir -e .
