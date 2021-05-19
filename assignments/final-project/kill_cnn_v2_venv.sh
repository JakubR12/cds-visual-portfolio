#!/usr/bin/env bash

VENVNAME=cnn_v2_venv
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME