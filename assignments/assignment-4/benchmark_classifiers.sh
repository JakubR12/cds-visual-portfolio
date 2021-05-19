#!/usr/bin/env bash

# make the environment
bash create_benchmark_class_venv.sh

source benchmark_class_venv/bin/activate

cd src

#run the scripts
python lr_mnist.py
python nn_mnist.py

cd ..

deactivate

# kill the environment
bash kill_benchmark_class_venv.sh