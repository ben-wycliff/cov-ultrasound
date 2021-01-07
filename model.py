# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 23:18:21 2021

@author: - Ben Wycliff Mugalu
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from matplotlib import pyplot

# Load data progressively
datagen = ImageDataGenerator()

# create train iteration
trainit = datagen.flow_from_directory("data/train", class_mode="categorical")
# create test iteration
testit = datagen.flow_from_directory("data/test", class_mode="categorical")

#inspecting batch
batchX, batchy = trainit.next()
print(f"batch shape = {batchX.shape}, min = {batchX.min()}, max = {batchX.max()}")

image = batchX[0].astype("uint8")
print(image.shape)
pyplot.imshow(image)
pyplot.show()

