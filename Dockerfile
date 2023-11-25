FROM ubuntu:22.04 AS build
RUN apt-get update && apt-get install -y curl
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
  bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniconda
ENV PATH=$PATH:/root/miniconda/bin
RUN apt-get -y install swig build-essential python3-dev
RUN apt-get -y install git

#COPY environment.yml /root/environment.yml
# install mamba environment
RUN --mount=type=cache,target=/root/miniconda/pkgs,target=/root/.conda/pkgs \
  mamba create -n FinRL3 python=3.10
# Make RUN commands use the new environment:
SHELL ["mamba", "run", "--no-capture-output", "-n", "FinRL3", "/bin/bash", "-c"]
RUN pip install jupyter jupyterlab
RUN python -m ipykernel install --user --name=FinRL3
RUN jupyter notebook --generate-config
RUN pip install jupyterlab_nvdashboard
COPY . /workspace
RUN pip install -e /workspace
COPY jupyter_server_config.json /root/.jupyter/jupyter_server_config.json
RUN mamba install -y -c conda-forge jupyterlab-git

FROM alfred/cs221_finalproject:latest AS config
#FROM ubuntu:22.04
#COPY --from=build /root/miniconda/envs/FinRL3 /root/miniconda/envs/FinRL3
# Update workspace without having to reinstall FinRL
COPY . /workspace
WORKDIR /workspace
RUN git config --global user.email "alfred.wechselberger@gmail.com"
RUN git config --global user.name "pongnguy"
RUN mkdir -p /root/.ssh
COPY id_rsa /root/.ssh
RUN ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
RUN chmod go-rwx /root/.ssh/id_rsa



ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "FinRL3", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]