import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_image
# import cv2
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
import json

# to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tf.config.set_visible_devices(cpu[0], 'CPU')

len(os.listdir("project dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"))

train_datagen = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, horizontal_flip=True,
                                   preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train = train_datagen.flow_from_directory(
    directory="project dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
    target_size=(255, 255), batch_size=32)

val = val_datagen.flow_from_directory(
    directory="project dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
    target_size=(255, 255), batch_size=32)

t_img, label = train.next()


def plotImage(img_arr, label):
    for im, l in zip(img_arr, label):
        plt.figure(figsize=(5, 5))
        plt.imshow(im)
        plt.show()


plotImage(t_img[:1], label[:1])

# BUILDING THE MODEL

from keras.layers import Dense, Flatten
from keras.models import Model, model_from_json
from keras.applications.vgg19 import VGG19
import keras

base_model = VGG19(input_shape=(255, 255, 3), include_top=False)
for layers in base_model.layers:
    layers.trainable = False
base_model.summary()

X = Flatten()(base_model.output)
X = Dense(units=38, activation='softmax')(X)
model = Model(base_model.input, X)
# model.summary()
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# EARLY STOPPING AND MODEL CHECKPOINT

from keras.callbacks import ModelCheckpoint, EarlyStopping

# early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1)

# model checkpoint
mc = ModelCheckpoint(filepath="best_model.h5", monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1,
                     save_best_only=True)
cb = [es, mc]
his = model.fit(train, steps_per_epoch=16, epochs=50, verbose=1, callbacks=cb, validation_data=val,
                validation_steps=16)
h = his.history
h.keys()
model.save("my_model")

# Plotting the accuracy and loss grpah
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c="red")
plt.title('acc vs vcc')
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'], c="red")
plt.title('loss vs vloss')
plt.show()

# Loading best model
from keras.models import model_from_json

model_json = model.to_json()
with open("best_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("best_model.h5")
print("Model Saved")

# load json and create model
json_file = open('best_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("best_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
# from keras.models import load_model
# model = load_model("best_model.h5")
# acc = model.evaluate_generator(val)[1]
# print(f"Model accuracy is = {acc*100}%")

# Predicting
ref = dict(zip(list(train.class_indices.values()), list(train.class_indices.keys())))
from keras_preprocessing.image.utils import load_img


def prediction(path):
    img = load_img(path, target_size=(255, 255))
    i = img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im, axis=0)
    pred = np.argmax(model.predict(img))
    print(f"The diseases belongs to {ref[pred]} ")


path = "project dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/Blueberry___healthy/0a0b8f78-df2d-4cfc-becf-cde10fa2766b___RS_HL 5487.JPG"
prediction(path)
