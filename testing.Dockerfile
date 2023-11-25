FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
  bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniconda && PATH=$PATH:$HOME/miniconda/bin
ENV PATH=$PATH:/root/miniconda/bin

ENTRYPOINT /bin/bash