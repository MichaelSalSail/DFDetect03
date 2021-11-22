import cv2
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16

def save_frame(video_path, output_dir=""):
    '''
    Saves the first frame of the video in the output_dir.
    
    Args:
        video_path: path of the video including the video name.
        output_dir: path to save the frame, optional parameter.
    
    Returns:
        full directory of the saved image.
    '''
    # creating a video capture object
    video_object = cv2.VideoCapture(video_path)

    # get the first frame of the video
    ret,frame = video_object.read()

    # save
    result=output_dir+"/"+"test_shades.png"
    cv2.imwrite(result, frame)
    return result

def detect_shades(image_dir, output_dir=""):
    '''
    Uses the default VGG16 model for image classification.
    VGG16 is able to detect 1000 object types in photos.
    We are focused on sunglasses classification.
    
    Args:
        image_dir: directory of input image.
        output_dir: location for the .txt file, optional parameter.
    
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
    # store all relevant information in a .txt file
    full_name=output_dir+"/"+"result.txt"
    file1 = open(full_name,"w")
    # put it all in a list then append to the .txt file
    result=list()
    result.append('Top 5 Object Detection Predictions\n')
    top_5_shades=False
    for i in range(0,5):
        result.append('%s (%.2f%%)\n' % (label[0][i][1], label[0][i][2]*100))
        # label 'n04356056' is 'sunglasses, dark glasses, shades'
        if (label[0][i][0]=='n04356056'):
            top_5_shades=True
    if (top_5_shades==True):
        result.append("There appears to be sunglasses. Larger glasses are hard to deepfake.")
    else:
        result.append("No sunglasses detected.")
    file1.writelines(result)
    file1.close()


# -----------------------------------------Look here-----------------------------------------
# The directories for the application repository are different from the DFDetect03 repository!
# Change the variables as you see fit.

pre1 = os.getcwd()
pre2 = "/data/deepfake-detection-challenge/test_videos"
video_path = pre1+pre2+"/adohdulfwb.mp4"

output_dir = "eyeblink data labels/data/temp"

frame_loc=save_frame(video_path, output_dir)
detect_shades(frame_loc, output_dir)