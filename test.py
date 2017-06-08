import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imread

if  __name__ == '__main__':
    img = imread('./data/images/tiny-imagenet-200/val/images/val_0.JPEG')
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((0, 32), 44, 30, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()