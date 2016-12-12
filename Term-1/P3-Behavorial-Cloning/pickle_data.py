### Import
import pickle
import os
import matplotlib.image as mpimg
import numpy as np

images = os.listdir("simulator-windows-64/Recorded Data/IMG/")

center_images = []

for idx, val in enumerate(images):
    # reading in an image
    if 'center' in images[idx]:
        image = mpimg.imread("simulator-windows-64/Recorded Data/IMG/" + images[idx])
        center_images.append(image)

features = np.array(center_images)
print(features.shape)

