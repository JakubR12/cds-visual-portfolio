#!/usr/bin/env bash

VENVNAME=edge_detection_venv
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME