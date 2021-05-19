#!/usr/bin/env bash


source cnn_v2_venv/bin/activate

cd src

#run the script

#dropuut
python VGG16_artist.py  -d 0.5 -is 64 -n "A2B_D5_IS64"

python VGG16_artist.py  -d 0.2 -is 64 -n "A2B_D2_IS64"

python VGG16_artist.py  -d 0.0 -is 64 -n "A2B_D0_IS64"


#image_size
python VGG16_artist.py -is 256 -n "A2B_D3_IS256"

python VGG16_artist.py -is 128 -n "A2B_D3_IS128"

python VGG16_artist.py -is 64 -n "A2B_D3_IS64"

python VGG16_artist.py -is 32 -n "A2B_D3_IS32"



cd ..

deactivate