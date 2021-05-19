#!/usr/bin/env bash

# make the environment
bash create_cnn_v2_venv.sh

source cnn_v2_venv/bin/activate

cd src

#run the scripts
python artist_classifier.py

cd ..

deactivate

# kill the environment
bash cnn_v2_venv.sh