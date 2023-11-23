FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && apt update
RUN apt-get install -y python3.10 python3.10-dev python3.10-distutils
RUN apt-get install -y swig build-essential python3-dev
RUN apt-get install -y python3-pip
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install gym elegantrl gputil pyfolio empyrical gpustat peewee AutoROM.accept-rom-license
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install finrl
COPY ./requirements.txt /root/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install -r /root/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install jupyterlab
COPY examples/ /root/
RUN apt-get -y install git
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install stable-baselines3
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install dill


ENTRYPOINT /bin/bash -c "jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser"