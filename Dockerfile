FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update

RUN pip install fire sentencepiece datasets pybind11

COPY llama_fast /code
WORKDIR /code

ENV TORCH_CUDA_ARCH_LIST="7.0+PTX"
RUN pip install -v --disable-pip-version-check --no-build-isolation --global-option="--cuda_ext" /code/apex_subset
RUN pip install /code/teamh_c_helper

CMD ["/code/run.sh"]