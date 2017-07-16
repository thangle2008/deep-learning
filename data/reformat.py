import os
import shutil


# For tiny-imagenet
def format_training_folder(folder):
    for root, dirnames, filenames in os.walk(folder):

        current_dir = root.split('/')[-1]

        if current_dir != 'images':
            continue

        for filename in filenames:
            src = os.path.join(root, filename)
            dst = os.path.join(root, '..', filename)
            shutil.move(src, dst)

        os.rmdir(root)


def distribute_images(folder, annotation_file):
    """
    Split a folder of images to a subdirectory per class
    based on an annotation file.
    """
    with open(annotation_file, 'rb') as f:
        for line in f:
            fname, label, lx, ly, tx, ty = line.split()

            fpath = os.path.join(folder, fname)
            subfolder = os.path.join(folder, label)

            # check if the subdirectory for this folder already exists
            if not os.path.isdir(subfolder):
                os.makedirs(subfolder)

            # move image to its subfolder
            shutil.move(fpath, subfolder)