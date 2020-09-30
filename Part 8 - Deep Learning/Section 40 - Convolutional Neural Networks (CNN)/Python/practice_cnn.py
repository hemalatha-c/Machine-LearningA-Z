#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 13:17:46 2020

@author: hemalatha

Convolutional Neural Network
"""
#import libraries
import tensorflow as tf
print(tf.__version__)

################################### PART -1 : DATA PREPROCESSING ##########
# pre processing training set

#create data augmentation obj
training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
#apply data augmentation on train set
training_set = training_datagen.flow_from_directory('dataset/training_set',
                                                    target_size = (64,64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

# pre processing test set

#create data augmentation object
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#apply data augmentation on test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'binary')

################################### PART -2: BUILD CNN ####################
#initialize CNN
cnn = tf.keras.models.Sequential()

#Step-1: Convolution => add conv layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=3,
                               activation='relu',
                               input_shape = [64,64,3]))
#Step 2: Max pooling: Add pooling layer
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = 2))

#Add second conv layer with max pooling
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size= 3, activation ='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = 2))

# Step 3: Flattening
cnn.add(tf.keras.layers.Flatten())

#Step 4: Add Fully connected layer
cnn.add(tf.keras.layers.Dense(units= 128, activation = 'relu'))

#Step 5: Output layer
cnn.add(tf.keras.layers.Dense(units= 1, activation = 'sigmoid'))

################################### PART -3 : COMPILE CNN #################
cnn.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])


################################### PART -4 : PREDICT  ####################
cnn.fit(x= training_set, validation_data= test_set, steps_per_epoch=len(training_set),
        validation_steps=len(test_set), epochs=25)

##Make single prediction
import numpy as np
#load test image
test_image = tf.keras.preprocessing.image.load_img(
    'dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64,64))

#convert image to array from PIL image format to array
test_image = tf.keras.preprocessing.image.img_to_array(test_image)

# Add this single image to batch, so that cnn recognizses image for prediction
test_image = np.expand_dims(test_image, axis = 0)

result = cnn.predict(test_image)

#get the class indices
training_set.class_indices

#acess result[batch][single ele in batch]
print("PREDICTION : ", result[0][0])
if result[0][0] == 1:
    print("Dog")
else:
    print("Cat")