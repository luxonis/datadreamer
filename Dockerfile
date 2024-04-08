## define a base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

## set working directory
WORKDIR /app

## instal
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install -y git
RUN git clone https://github.com/luxonis/datadreamer.git -b main

RUN cd datadreamer && pip install .

## define image execution
ENTRYPOINT ["datadreamer"]