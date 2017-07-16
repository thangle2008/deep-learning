#!/usr/bin/env bash

wget 'https://drive.google.com/uc?export=download&id=0B1jo1AcPnGtFSnU5cUVEUi1HalU' -O bird_342x256.zip
mkdir -p images
unzip bird_342x256.zip -d ./images
rm -rf bird_342x256.zip