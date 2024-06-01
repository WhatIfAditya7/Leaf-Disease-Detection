from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_image
import numpy as np
import os
from keras_preprocessing.image.utils import load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.layers import Dense, Flatten
from keras.models import Model, model_from_json
from keras.applications.vgg19 import VGG19
import keras

def get_model():
    base_model = VGG19(input_shape=(255, 255, 3), include_top=False)
    for layers in base_model.layers:
        layers.trainable = False
    base_model.summary()

    X = Flatten()(base_model.output)
    X = Dense(units=38, activation='softmax')(X)
    model = Model(base_model.input, X)
    model.summary()
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model

# to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = keras.models.load_model("my_model")
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

train_datagen = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, horizontal_flip=True,
                                   preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train = train_datagen.flow_from_directory(
    directory="project dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
    target_size=(255, 255), batch_size=32)
ref = dict(zip(list(train.class_indices.values()), list(train.class_indices.keys())))

def prediction(path):
    img = load_img(path, target_size=(255, 255))
    i = img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im, axis=0)
    pred = np.argmax(model.predict(img))
    print(f"The diseases belongs to {ref[pred]} ")
    return ref[pred]


path = "project dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/Grape___Esca_(Black_Measles)/0e944041-c132-49cb-bd8c-0224c0f06cad___FAM_B.Msls 1241.JPG"
prediction(path)