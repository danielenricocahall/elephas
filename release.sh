#!/bin/bash

# remove old wheels
sudo rm -rf dist/*

# Build Python 3 wheels for current version
poetry build -f wheel

# Upload to PyPI with Poetry.
poetry publish