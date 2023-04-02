FROM nvidia/cuda:11.4.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3 python3-pip gfortran build-essential libhdf5-openmpi-dev openmpi-bin pkg-config libopenmpi-dev openmpi-bin libblas-dev liblapack-dev libpnetcdf-dev git python-is-python3 gnuplot
RUN pip3 install numpy matplotlib h5py scipy sympy yt
ENV USER=jenkins
ENV LOGNAME=jenkins
