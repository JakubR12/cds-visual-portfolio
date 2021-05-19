#!/usr/bin/env bash

# make the environment
bash create_cnn_venv.sh

source cnn_venv/bin/activate

cd src

#run the script
python cnn-artists.py

cd ..

deactivate

# kill the environment
bash kill_cnn_venv.sh