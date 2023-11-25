FROM ubuntu:22.04 AS build
RUN apt-get update && apt-get install -y curl
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
  bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniconda
ENV PATH=$PATH:/root/miniconda/bin
#RUN apt-get install -y software-properties-common
RUN apt-get -y install swig build-essential python3-dev
RUN apt-get -y install git

#COPY environment.yml /root/environment.yml
# install mamba environment
RUN --mount=type=cache,target=/root/miniconda/pkgs,target=/root/.conda/pkgs \
  mamba create -n FinRL3 python=3.10
# Make RUN commands use the new environment:
SHELL ["mamba", "run", "--no-capture-output", "-n", "FinRL3", "/bin/bash", "-c"]
#RUN --mount=type=cache,target=/root/.cache/pip \
#  pip install gym elegantrl gputil pyfolio empyrical gpustat peewee AutoROM.accept-rom-license
#RUN --mount=type=cache,target=/root/.cache/pip \
#  pip install finrl
#RUN --mount=type=cache,target=/root/.cache/pip \
#  pip install jupyterlab
#RUN --mount=type=cache,target=/root/.cache/pip \
#  pip install stable-baselines3
#COPY requirements.txt /root/requirements.txt
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install -r /root/requirements.txt
RUN pip install jupyter jupyterlab
#RUN pip install finrl jupyter jupyterlab stable-baselines3 gym elegantrl gputil pyfolio empyrical gpustat peewee AutoROM.accept-rom-license
RUN python -m ipykernel install --user --name=FinRL3
RUN jupyter notebook --generate-config
# TODO triple <<< messes with syntax highlighting
#RUN jupyter notebook password <<< $'Cc17931793\nCc17931793\n'
#RUN echo 'Cc17931793\nCc17931793\n' | jupyter notebook password
# Copy over precomputed password hash
#COPY jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json
RUN pip install jupyterlab_nvdashboard
#RUN mamba install cuda
#RUN apt-get install -y nvidia-docker2
COPY . /workspace
RUN pip install -e /workspace
#RUN pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
#COPY jupyter_notebook_config.json /root/.jupyter/jupyter_notebook_config.json
COPY jupyter_server_config.json /root/.jupyter/jupyter_server_config.json
#RUN jupyter notebook password <<< $'Cc17931793\nCc17931793\n'
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
#CMD ["jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser"]
#ENTRYPOINT ["/root/miniconda/envs/FinRL3/bin/jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
#ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
