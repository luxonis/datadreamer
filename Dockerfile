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

RUN cd datadreamer && pip install .

## define image execution
ENTRYPOINT ["datadreamer"]
