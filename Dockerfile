FROM jupyter/datascience-notebook:latest
RUN pip install --upgrade pip
# nothing here yet

ENTRYPOINT "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='$TOKEN'"