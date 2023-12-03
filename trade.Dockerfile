FROM cs221:base
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
#FROM ubuntu:22.04
#COPY --from=build /root/miniconda/envs/FinRL3 /root/miniconda/envs/FinRL3
# Update workspace without having to reinstall FinRL
COPY ./examples /workspace/examples
#RUN --mount=type=cache,target=/root/.cache/pip \
#    pip install chardet cchardet

WORKDIR /workspace/examples
ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "FinRL3", "python3", "load_FinRL_PaperTrading_Demo_refactored.py", "PKUBMLMM7EIUT4NT5C3O", "76if73jMuNNeEyru8KnjADEXf0VoDGZlneKrimEU", "https://paper-api.alpaca.markets", "PKUBMLMM7EIUT4NT5C3O", "76if73jMuNNeEyru8KnjADEXf0VoDGZlneKrimEU", "https://paper-api.alpaca.markets"]