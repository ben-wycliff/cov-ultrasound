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
from keras.utils import to_categorical

# Load data progressively
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    featurewise_center=True,
    featurewise_std_normalization=True,
    height_shift_range=0.1, 
    width_shift_range=0.1, 
    horizontal_flip=True, 
    vertical_flip=True, 
    rotation_range=10)

# create train iteration
trainit = datagen.flow_from_directory("data/train", target_size=(224, 224), class_mode="categorical")
# create test iteration
testit = datagen.flow_from_directory("data/test", target_size=(224, 224), class_mode="categorical")

#inspecting batch
batchX, batchy = trainit.next()
print(f"batch shape = {batchX.shape}, min = {batchX.min()}, max = {batchX.max()}")

# inspecting image
# image = batchX[0].astype("uint8")
# print(image.shape)
# pyplot.imshow(image)
# pyplot.show()


