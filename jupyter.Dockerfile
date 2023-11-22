FROM jupyter/datascience-notebook:latest
#RUN pip install --upgrade pip
COPY ./ ./home/jovyan/work
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
  bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniconda
RUN mamba install python=3.10

ENTRYPOINT "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='$TOKEN'"