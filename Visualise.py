import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import tensorflow as tf
from tensorflow import keras
from Dataset import train_dataset,test_dataset,val_dataset

# Visualising Dataset
def Visualise_dataset(dataset):
    images = next(iter(dataset)).numpy()
    fig = plt.figure(figsize=(12,12))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)
    random_images = images[np.random.choice(np.arange(images.shape[0]), 12, replace=False)]
    for ax, image in zip(grid, random_images):
        image = image * 255.0
        ax.imshow(image.astype(np.uint8))
    plt.show()

Visualise_dataset(train_dataset)

