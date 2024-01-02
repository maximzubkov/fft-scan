FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential \
                       ca-certificates \
                       wget \
                       curl \
                       unzip \
                       ssh \
                       git \
                       vim \
                       jq

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.11-full
RUN apt-get install -y python3.11-dev
RUN apt-get clean
RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python -m pip install --upgrade pip

WORKDIR /pscan

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN pip cache purge