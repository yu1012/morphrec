#!bin/bash

docker build -t morphrec:base .
docker run -it --ipc=host --pid=host --name morphrec --gpus all -v $(pwd):/workspace morphrec:base
