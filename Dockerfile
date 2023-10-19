FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN apt-get update

# conda
RUN mkdir -p /usr/local/miniconda3
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /usr/local/miniconda3/miniconda.sh
RUN bash /usr/local/miniconda3/miniconda.sh -b -u -p /usr/local/miniconda3
ENV PATH="$PATH:/usr/local/miniconda3/bin"
RUN rm -rf /usr/local/miniconda3/miniconda.sh

# pytorch 2.1.0
RUN pip uninstall -y torch torch-tensorrt torchdata torchtext torchvision
RUN wget -q https://github.com/pytorch/pytorch/releases/download/v2.1.0/pytorch-v2.1.0.tar.gz -O /usr/local/pytorch-v2.1.0.tar.gz
RUN tar -xzf /usr/local/pytorch-v2.1.0.tar.gz -C /usr/local/
RUN conda install -y cmake ninja mkl mkl-include 
RUN conda install -y -c pytorch magma-cuda121
RUN export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
WORKDIR /usr/local/pytorch-v2.1.0
RUN python /usr/local/pytorch-v2.1.0/setup.py install

# other libs
RUN pip install fire sentencepiece datasets pybind11
RUN apt-get install bc

# build csrc
COPY deps /deps
ENV TORCH_CUDA_ARCH_LIST="7.0+PTX"
RUN pip install -v --disable-pip-version-check --no-build-isolation --global-option="--cuda_ext" /deps/apex_subset
RUN pip install /deps/teamh_c_helper

# copy llama code
COPY llama_fast /code
WORKDIR /code

CMD ["/code/run.sh"]