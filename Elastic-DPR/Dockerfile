FROM  nvidia/cuda:11.0-runtime-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3.7-distutils python3-pip curl wget git pkg-config cmake swig

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Set default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN update-alternatives --set python3 /usr/bin/python3.7

RUN apt-get install libpoppler-cpp-dev pkg-config -y --fix-missing

RUN pip3 install numpy scipy Cython

# Install PyTorch for CUDA 11
RUN pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
# Install faiss separately as building latest versions can cause trouble with swig
RUN pip3 install faiss-cpu==1.6.3

# install as a package
COPY requirements.txt /app

WORKDIR /app

# install as a package
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# copy code
COPY . .

EXPOSE 8000

# cmd for running the API
CMD sleep 60 && python3 index.py && uvicorn app:app
