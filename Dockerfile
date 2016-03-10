FROM continuumio/anaconda3:latest

MAINTAINER Prasad Chodavarapu <chprasad@gmail.com>

RUN apt-get update --fix-missing && \
    apt-get install -y python3-dev g++ libopenblas-dev && \
    conda update -y conda numpy scipy pip nose pillow && \
    conda install Theano

RUN git clone https://github.com/rakeshvar/theanet.git && \
    git clone https://github.com/TeluguOCR/banti_telugu_ocr.git && \
    cd theanet && python3 setup.py install && \
    apt-get install -y ghostscript imagemagick
    
ENV THEANO_FLAGS 'floatX=float32'

WORKDIR /banti_telugu_ocr

