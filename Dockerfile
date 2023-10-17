FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN apt-get update

RUN pip install fire sentencepiece datasets pybind11

COPY llama_fast /code
WORKDIR /code

ENV TORCH_CUDA_ARCH_LIST="7.0+PTX"
RUN pip install -v --disable-pip-version-check --no-build-isolation --global-option="--cuda_ext" /code/apex_subset
RUN pip install /code/teamh_c_helper

CMD ["/code/run.sh"]