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

# def vgg_block(layer_in, n_filters, n_conv):
#     for i in range(n_conv):
#         layer_in = Conv2D(n_filters, (3,3), padding="same", activation="relu")(layer_in)
#     layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
#     return layer_in

# visible = Input(shape=(256, 256, 3))
# layer = vgg_block(visible, 64, 2)
# model = Model (inputs=visible, outputs=layer)
# model.summary()
# plot_model(model, show_shapes=True, to_file="vgg_block.jpg")

# Load data progressively
datagen = ImageDataGenerator()

# create train iteration
trainit = datagen.flow_from_directory("data/train", class_mode="categorical")
# create test iteration
testit = datagen.flow_from_directory("data/test", class_mode="categorical")

#inspecting batch
batchX, batchy = trainit.next()
print(f"batch shape = {batchX.shape}, min = {batchX.min()}, max = {batchX.max()}")