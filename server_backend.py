from flask import Flask, render_template,request
from flask_wtf import FlaskForm
from wtforms import FileField
import base64
import moviepy.editor as mp
import math


import os, sys, time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageStat
import sys
from torchvision.transforms import Normalize
import torch.nn as nn
import torchvision.models as models
#import rcmd


result = {}

# Setting up the model reuirements

# get the directory of the models
cwd = os.getcwd()
sys.path.insert(0,cwd + "/imports/blazeface")
sys.path.insert(0,cwd +  "/imports/inference")

from blazeface import BlazeFace


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load blazeface training weights
facedet = BlazeFace().to(device)
facedet.load_weights(cwd + "/imports/blazeface/blazeface.pth")
facedet.load_anchors(cwd + "/imports/blazeface/anchors.npy")
_ = facedet.train(False)


from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor

frames_per_video = 150

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)
input_size =224 


mean = [0.43216, 0.394666, 0.37645]
std = [0.22803, 0.22145, 0.216989]
normalize_transform = Normalize(mean,std)

class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)
        self.fc = nn.Linear(2048, 1)

checkpoint = torch.load(cwd + "/imports/inference/resnext.pth", map_location=device)

model = MyResNeXt().to(device)
model.load_state_dict(checkpoint)
_ = model.eval()

del checkpoint



### MODEL FUNCTIONS

def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized

# here we use opencv to make the fram square 

def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)
        self.fc = nn.Linear(2048, 1)

checkpoint = torch.load(cwd + "/imports/inference/resnext.pth", map_location=device)

model = MyResNeXt().to(device)
model.load_state_dict(checkpoint)
_ = model.eval()

del checkpoint


def predict_on_video(video_path, batch_size):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)
        face_extractor.keep_only_best_face(faces)
        
        if len(faces) > 0:
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    #print("Before resize")
                    #plt.imshow(face, interpolation='nearest')
                    #plt.show()
                    #print("After Resize")
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = make_square_image(resized_face)
                    #print("Before any preproccessing ")
                    #plt.imshow(resized_face, interpolation='nearest')
                    #plt.show()
                    #brightness_value = int(math.ceil(detect_brightness_algo2(resized_face)))
                    
                    #if brightness_value < 125:
                    #    increase_by = 125 - brightness_value
                    #    print("increase_by :" + str(increase_by))
                    #    resized_face = increase_brightness(resized_face,increase_by)
                    #    print("AFter adding brightness")
                    #    plt.imshow(resized_face, interpolation='nearest')
                    #    plt.show()     
                    #print(brightness_value)

                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))

            if n > 0:
                x = torch.tensor(x, device=device).float()
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)
                with torch.no_grad():
                    y_pred = model(x)
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    return y_pred[:n].mean().item()

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5


### END OF MODEL FUNCTIONS

# split videos

# predict videos

def splice_video(filename, clip_length):
    output_data=[]
    
    # get video from directory
    video = mp.VideoFileClip(filename)
    # get length of video
    video_duration = video.duration
    # get number of clips that need to be made
    segments = math.ceil(video_duration/clip_length)
    # temp variable to hold clips before writing
    clips = []
    # splice video
    for i in range(segments):
        if clip_length*(i+1) > video_duration:
            clip = video.subclip(i*clip_length, video_duration)
            clips.append(clip)
        else:
            clip = video.subclip(i*clip_length, clip_length*(i+1))
            clips.append(clip)
    # write clips
    for i in range(len(clips)):
        write_name = "clip" + str(i) +".mp4"
        clips[i].write_videofile(write_name)
        output_data.append(write_name)
    # close video
    return output_data
    video.close()




app = Flask(__name__);

@app.route('/', methods=['GET','POST'])
def welcome():
    return "Hello World!"

@app.route('/upload/', methods=['POST'])
def upload():
	content = request.get_json()
	length = request.content_length;
	if length>20000000:
		result["error"]="files size is too large . Maximum size is 20 mb";
		return result
	else:
		#first we decode the file in base64
		video_data=content['data']
		file_handler = open("temp.mp4", "wb")
		file_handler.write(base64.b64decode(video_data))
		file_handler.close()
		spliced_videos = splice_video("temp.mp4",20);
		split_result={}
		for i in range(0,len(spliced_videos)):
			split_result[i]=predict_on_video(spliced_videos[i],frames_per_video)

		## get prediction
		#video_prediction=predict_on_video("temp.mp4",frames_per_video);

		##
		result["preditction"]=split_result;
		return result;
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)


