## define a base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

## set working directory
WORKDIR /app

## instal
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y git

## Define a build argument for the branch, defaulting to "main"
ARG BRANCH=main

## Clone the repository with the specified branch
RUN git clone --branch ${BRANCH} https://github.com/luxonis/datadreamer.git

## Create a non-root user and switch to that user
RUN adduser --disabled-password --gecos "" non-root && \
    chown -R non-root:non-root /app

## Switch to the non-root user
USER non-root

## Install the Python package as the non-root user
RUN cd datadreamer && pip install .

## Set PATH for the installed executable
ENV PATH="/home/non-root/.local/bin:/usr/local/bin:$PATH"


## define image execution
ENTRYPOINT ["datadreamer"]
