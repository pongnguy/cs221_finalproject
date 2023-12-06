FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
  bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniconda
ENV PATH=$PATH:/root/miniconda/bin
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
RUN apt-get -y install swig build-essential python3-dev
RUN apt-get -y install git

#COPY environment.yml /root/environment.yml
# install mamba environment
RUN --mount=type=cache,target=/root/miniconda/pkgs,target=/root/.conda/pkgs \
  mamba create -n FinRL4 python=3.10
# Make RUN commands use the new environment:
SHELL ["mamba", "run", "--no-capture-output", "-n", "FinRL4", "/bin/bash", "-c"]
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install jupyter jupyterlab
RUN python -m ipykernel install --user --name=FinRL4
RUN jupyter notebook --generate-config
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install jupyterlab_nvdashboard
RUN mamba install -y -c conda-forge jupyterlab-git
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install docker dill optuna chardet cchardet
COPY jupyter_server_config.json /root/.jupyter/jupyter_server_config.json
COPY . /workspace
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e /workspace
# hack to copy over custom diskcache, think about how to address for dev environment
#COPY /python-diskcache /python-diskcache
#RUN pip install /python-diskcache
