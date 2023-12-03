FROM cs221:base
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
#FROM ubuntu:22.04
#COPY --from=build /root/miniconda/envs/FinRL3 /root/miniconda/envs/FinRL3
# Update workspace without having to reinstall FinRL
COPY . /workspace
#COPY ./examples /workspace/examples
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install chardet cchardet

WORKDIR /workspace/examples
ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "FinRL3", "python3", "FinRL_PaperTrading_Demo_refactored.py"]