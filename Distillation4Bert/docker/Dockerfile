FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch

WORKDIR /workspace
COPY . general_distiller/
RUN cd general_distiller/ && \
    python3 -m pip install -r requirements.txt

CMD ["/bin/bash"]