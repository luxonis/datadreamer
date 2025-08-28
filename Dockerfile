## define a base image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

## set working directory
WORKDIR /app

## noninteractive apt + branch arg
ARG DEBIAN_FRONTEND=noninteractive
## Define a build argument for the branch, defaulting to "main"
ARG BRANCH=main

## deps + shallow clone, then drop git and clean apt caches
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsm6 libxext6 git \
 && git clone --branch "${BRANCH}" --depth 1 https://github.com/luxonis/datadreamer.git /app/datadreamer \
 && apt-get purge -y --auto-remove git \
 && rm -rf /var/lib/apt/lists/*

## Create a non-root user and switch to that user
RUN adduser --disabled-password --gecos "" non-root && \
    chown -R non-root:non-root /app

## Switch to the non-root user
USER non-root

## Install the Python package as the non-root user (no pip cache), then drop sources
RUN pip install --no-cache-dir /app/datadreamer && rm -rf /app/datadreamer

## Set PATH for the installed executable
ENV PATH="/home/non-root/.local/bin:/usr/local/bin:$PATH"

## Define image execution
ENTRYPOINT ["datadreamer"]
