#!/usr/bin/env bash

# make the environment
bash create_edge_detection_venv.sh

source edge_detection_venv/bin/activate

cd src

#run the script
python edge_detection.py

cd ..

deactivate

# kill the environment
bash kill_edge_detection_venv.sh