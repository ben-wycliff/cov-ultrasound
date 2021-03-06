# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 23:18:21 2021

@author: - Ben Wycliff Mugalu
"""

from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
#======================== Load data progressively =============================
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    samplewise_center=True,
    samplewise_std_normalization=True,
    height_shift_range=0.1, 
    width_shift_range=0.1, 
    horizontal_flip=True, 
    vertical_flip=True, 
    rotation_range=10)

# create train iteration
trainit = datagen.flow_from_directory("data/train", target_size=(224, 224), class_mode="categorical", batch_size=32)
# create test iteration
testit = datagen.flow_from_directory("data/test", target_size=(224, 224), class_mode="categorical", batch_size=32)

#inspecting batch
batchX, batchy = trainit.next()
print(f"batch shape = {batchX.shape}, min = {batchX.min()}, max = {batchX.max()}")

# inspecting image
# image = batchX[0].astype("uint8")
# print(image.shape)
# pyplot.imshow(image)
# pyplot.show()

# ============================ Creating Model ===================================
def define_model():
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
        
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation="relu")(flat1)
    dropout = Dropout(0.5)(class1)
    normalized = BatchNormalization()(dropout)
    output = Dense(3, activation="softmax")(normalized)
    model = Model(inputs=model.inputs, outputs=output)
    model.summary()
    # compile model
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_test_model(train_it, test_it):
    model = define_model()
    model.fit(train_it, epochs=128)
    loss = model.evaluate(test_it, steps=24)
    print(loss)
    return model
train_test_model(trainit, testit)