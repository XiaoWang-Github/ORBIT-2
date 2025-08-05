#Dockerfile
FROM nvcr.io/nvidia/pytorch:25.05-py3


RUN pip install ninja && \
    pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@v0.0.30 && \
    pip install scikit-image && \
    pip install matplotlib

COPY super-res-torchlight /opt/super-res-torchlight \

RUN cd /opt/super-res-torchlight && \
  pip install -e .
