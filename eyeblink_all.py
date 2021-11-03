import os
import cv2
import dlib
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from scipy.spatial import distance as dist
from imutils import face_utils
import face_recognition
from PIL import Image


# Prepare images.
def images_ready(all_imgs,folder_name):
    result=[]
    for i in all_imgs:
        # load an image from file
        get_file=folder_name+str(i)
        image = load_img(get_file, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        result.append(image)
    return result

def more_tests(model, folder_dir):
    # param: , name_csv
    # extra_test = pd.read_csv(name_csv, header=0)
    # fast_blink_names=extra_test['Name']
    fast_blink_names = sorted([x for x in os.listdir(folder_dir)])
    fast_blink_names=np.array(fast_blink_names)
    fast_blink_names=images_ready(fast_blink_names, folder_dir+"/")
    regular_3=np.array(fast_blink_names)
    good_shape_3=np.squeeze(regular_3)
    result_vgg16_2= model.predict(good_shape_3)
    count_1=0
    for i in range(0,len(result_vgg16_2)):
        if result_vgg16_2[i]>0.5:
            count_1+=1
    print("The model predicts",(count_1/150)*100,"% of the frames are eyes open.")
    print("The model predicts",100-((count_1/150)*100),"% of the frames are eyes closed.")

def full_model_run():
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # don't train upper layers
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.0001),metrics=['accuracy'])
    model.load_weights("data/Weights/all_weights_eyeblink.ckpt")