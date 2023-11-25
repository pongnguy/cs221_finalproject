FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
  bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniconda
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN $HOME/miniconda/bin/mamba install python=3.10
COPY ./requirements.txt /home/jovyan/work/requirements.txt
RUN apt-get -y install swig build-essential python3-dev
RUN --mount=type=cache,target=/root/.cache/pip \
  $HOME/miniconda/bin/pip install gym elegantrl gputil pyfolio empyrical gpustat peewee AutoROM.accept-rom-license
RUN --mount=type=cache,target=/root/.cache/pip \
  $HOME/miniconda/bin/pip install finrl
RUN --mount=type=cache,target=/root/.cache/pip \
  $HOME/miniconda/bin/pip install -r /home/jovyan/work/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
  $HOME/miniconda/bin/pip install jupyterlab
COPY examples/ /root/
RUN apt-get -y install git
RUN --mount=type=cache,target=/root/.cache/pip \
  $HOME/miniconda/bin/pip install stable-baselines3
# TODO activate miniconda environment


ENTRYPOINT /bin/bash -c "$HOME/miniconda/bin/jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser"