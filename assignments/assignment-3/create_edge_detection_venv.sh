#!/usr/bin/env bash

VENVNAME=edge_detection_venv

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter

python -m ipykernel install --name=$VENVNAME --user

test -f requirements.txt && pip install -r requirements.txt

pip freeze | grep -v "pkg-resources" > requirements.txt

deactivate
echo "build $VENVNAME"