FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
  bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniconda
RUN $HOME/miniconda/bin/mamba install python=3.10
COPY ./requirements.txt /home/jovyan/work/requirements.txt
RUN $HOME/miniconda/bin/pip install -r /home/jovyan/work/requirements.txt

ENTRYPOINT /bin/bash