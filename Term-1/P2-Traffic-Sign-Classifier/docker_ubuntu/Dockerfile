#FROM ubuntu:trusty
FROM continuumio/anaconda3:latest


RUN apt-get update

# Helper libraries
RUN apt-get install -y \
       git \
       man \
       vim \
       libgtk2.0-0

RUN conda install -y \
          -c https://conda.anaconda.org/menpo opencv3

RUN conda install -y \
          -c conda-forge tensorflow

RUN conda install -y \
          jupyter \
          scikit-learn

# Helper libraries
#RUN apt-get install -y \
#       git \
#       man \
#       python-dev \
#       python-pip \
#       vim

EXPOSE 5000-65535

CMD ["/bin/bash"]
