FROM dzcr/base
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
#FROM ubuntu:22.04
#COPY --from=build /root/miniconda/envs/FinRL3 /root/miniconda/envs/FinRL3
# Update workspace without having to reinstall FinRL
RUN apt-get install -y parallel
#RUN --mount=type=cache,target=/var/lib/apt/lists \
#    --mount=type=cache,target=/var/cache/apt \
#  apt-get install -y apt-transport-https ca-certificates gnupg curl && \
#  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
#  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
#  apt-get update && apt-get install -y google-cloud-cli
#RUN mamba install -y google-cloud-storage
#COPY . /workspace
#RUN --mount=type=secret,id=gcloud.json \
#  #gcloud auth activate-service-account --key-file=/run/secrets/gcloud.json && \
#  cp /run/secrets/gcloud.json /workspace/gcloud.json
#  #gcloud auth application-default login
#ENV GOOGLE_APPLICATION_CREDENTIALS="/workspace/gcloud.json"
#COPY ./examples /workspace/examples
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install chardet cchardet


WORKDIR /workspace/examples
ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "FinRL3", "/bin/bash", "-c", "parallel -j $N_PROCESSES -u python3 ::: $(seq $N_PROCESSES | xargs -I{} echo 'Stock_NeurIPS2018_optuna.py') && echo 'spawned $N_PROCESSES of Stock_NeurIPS2018_optuna.py'"]
#ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "FinRL3", "/bin/bash", "-c", "parallel -j $N_PROCESSES -u python3 ::: $(seq $N_PROCESSES | xargs -I{} echo 'FinRL_PaperTrading_Demo_refactored.py')"]
