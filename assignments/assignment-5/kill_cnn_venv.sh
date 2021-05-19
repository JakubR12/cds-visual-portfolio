#!/usr/bin/env bash

VENVNAME=cnn_venv
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME