#!/usr/bin/env bash

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -O tiny-imagenet-200.zip
mkdir -p images
unzip tiny-imagenet-200.zip -d ./images
rm -rf tiny-imagenet-200.zip
python -c "from reformat import format_training_folder; format_training_folder('./images/tiny-imagenet-200/train')"
python -c "from reformat import distribute_images; distribute_images('./images/tiny-imagenet-200/val/images',
    './images/tiny-imagenet-200/val/val_annotations.txt')"
