import os
import numpy as np
import face_recognition
from deepface import DeepFace
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


def images_ready(all_imgs,folder_name):
    '''
    Convert the images to proper format to run predictions
    on a model.
    
    Args:
        all_imgs: names of all images in a directory.
        folder_dir: directory of all_imgs.
        
    Returns:
        List of images in array format.
    '''
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
    '''
    Runs a prediction on model.
    Prints the binary classification of the image in directory folder_dir.
    
    Args:
        model: transfer learn VGG16 model.
        folder_dir: directory of .png files.
        
    Returns:
        1 if the eyes are open, 0 if the eyes are closed.
    '''
    fast_blink_names = ["p.png"]
    fast_blink_names=np.array(fast_blink_names)
    fast_blink_names=images_ready(fast_blink_names, folder_dir+"/")
    regular_3=np.array(fast_blink_names)
    good_shape_3=np.squeeze(regular_3, axis=0)
    result_vgg16_2= model.predict(good_shape_3)
    count_1=0
    for i in range(0,len(result_vgg16_2)):
        if result_vgg16_2[i]>0.5:
            count_1=1
    if count_1==1:
        print("Eyes open.")
    else:
        print("Eyes closed.")
    return count_1

def get_model():
    '''
    Creates a model using VGG16.
    
    Args:
        None
        
    Returns:
        VGG16 model for binary classification. Upper layers
        are frozen for transfer learning.
    '''
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
    return model

def save_crop(input_image, file_name, destination):
    '''
    Crop an image to only contain a persons face.
    
    Args:
        input_image: name of the file to be cropped.
        file_name: name of the new cropped image.
        destination: directory in which pictures are located.
        
    Returns:
        None
    '''
    full_1=destination+file_name
    full_2=destination+input_image
    input_image=face_recognition.load_image_file(full_2)
    locations = face_recognition.face_locations(input_image)
    for x in locations:
        top, right, bottom, left = x
        face_image = input_image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(full_1)

def detect_beard(image_dir):
    '''
    Detects an adult male from image.
    
    Args:
        image_dir: directory of input image.
        
    Returns:
        None
    '''
    obj = DeepFace.analyze(img_path = image_dir, 
                           actions = ['age', 'gender'],
                           enforce_detection=False)
    print("   Age:", obj["age"])
    print("Gender:", obj["gender"])
    if (obj["age"]>=20 and obj["gender"]=='Man'):
        print("There appears to be an adult male. Beards are hard to deepfake.")

def detect_shades(image_dir):
    '''
    Uses the default VGG16 model for image classification.
    VGG16 is able to detect 1000 object types in photos.
    We are focused on sunglasses classification.
    
    Args:
        image_dir: directory of input image.
    
    Returns:
        None
    '''
    # resizes an image to required VGG16 dimensions.
    image = load_img(image_dir, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # load the model
    model = VGG16()
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # print the top 5 highest probabilities
    print('Top 5 Object Detection Predictions')
    print('%s (%.2f%%)' % (label[0][0][1], label[0][0][2]*100))
    print('%s (%.2f%%)' % (label[0][1][1], label[0][1][2]*100))
    print('%s (%.2f%%)' % (label[0][2][1], label[0][2][2]*100))
    print('%s (%.2f%%)' % (label[0][3][1], label[0][3][2]*100))
    print('%s (%.2f%%)' % (label[0][4][1], label[0][4][2]*100))
    # label 'n04356056' is 'sunglasses, dark glasses, shades'
    if(label[0][0][0]=='n04356056' or label[0][1][0]=='n04356056'\
       or label[0][2][0]=='n04356056' or label[0][3][0]=='n04356056'\
       or label[0][4][0]=='n04356056'):
        print("There appears to be sunglasses. Larger glasses are hard to deepfake.")
