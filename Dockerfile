FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update

RUN pip install fairscale fire sentencepiece
RUN pip install datasets

COPY llama_fast /code
WORKDIR /code

RUN pip install -v --disable-pip-version-check --no-build-isolation --global-option="--cuda_ext" /code

CMD ["/code/run.sh"]