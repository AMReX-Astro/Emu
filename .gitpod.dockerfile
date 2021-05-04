FROM gitpod/workspace-full

RUN sudo apt-get update && sudo apt-get install -y gfortran libopenmpi3 libopenmpi-dev && sudo rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="/workspace/Emu/Scripts/visualization:$PYTHONPATH"

