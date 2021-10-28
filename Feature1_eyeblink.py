#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import dlib
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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


# In[2]:


# Use the code from Chukwudi's notebook... save the plt.show() from the testing set as images for this notebook training set
# Eyes: 0 is closed, 1 is open.
# It dataset is ~65% open, 35% closed.
eyes_df = pd.read_csv('Feature1_labels.csv', header=0)
eyes_df.groupby('Eyes')['Eyes'].count()


# In[3]:


eyes_df


# In[4]:


# Train test split.
X_train, X_test, y_train, y_test = train_test_split(
 eyes_df['Name'],
 eyes_df['Eyes'],
 test_size=0.2,
 random_state=180,
 shuffle=True)
y_train.groupby(y_train).count()


# In[5]:


# Use blazeface to detect face + landmarks
print(X_train.iloc[0])
print(len(X_train))


# In[6]:


for i in X_train:
    print(i)


# In[7]:


X_train


# In[8]:


# Prepare images.
def images_ready(all_imgs):
    result=[]
    for i in all_imgs:
        # load an image from file
        get_file="Feature1_data/"+str(i)
        image = load_img(get_file, target_size=(432, 288))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        result.append(image)
    return result

X_train=images_ready(X_train)
X_test=images_ready(X_test)


# In[9]:


# Transfer learn VGG16 to detect eye blinks... binary classification.
# load model without classifier layers
model = VGG16(include_top=False, input_shape=(432, 288, 3))
# don't train upper layers
for layer in model.layers:
    layer.trainable = False
# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(256, activation='relu')(flat1)
output = Dense(2, activation='sigmoid')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize
model.summary()


# In[10]:


model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['accuracy'])


# In[11]:


regular=np.array(X_test)
good_shape=np.squeeze(regular)
print(regular.shape)
print(good_shape.shape)


# In[12]:


import tensorflow as tf
y_test = tf.one_hot(y_test, depth=2)


# In[13]:


from tensorflow.keras import backend as K
K.clear_session()


# In[ ]:


model.fit(good_shape, y_test, verbose=0)


# In[ ]:


# Preprocess images and get results. Break video into still frames.
np.shape(new_X_test)

