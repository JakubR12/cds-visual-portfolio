#!/usr/bin/env bash

VENVNAME=benchmark_class_venv
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME