# Deepfake Detection - Senior Capstone  

1. DeepFace - face detection from Facebook. After looking through
a face, it can describe the persons: gender, age, race, and emotion.
It may prove useful later in our project.  

2. face_recognition - straightforward tools for curating our dataset. 
Find the faces of people in images. Crop the images to return an image
of each face **OR** place bounding boxes on all faces in the image. I
didn't test it out yet, but it also can "Recognize faces in a video file
and write out a new video file."  

3. VGG16 - the model requires the images to be RGB and resized to 224*224.
This model is straightforward and used in a couple online transfer learning 
tutorials for integrated feature extraction.  

## Installation  

For face_recognition, you need to run from Linux. If you are using a windows
machine, install WSL 2. From there, run the commands:  

Make sure you have cmake  
'cmake --version'  
If not, do  
'sudo snap install cmake'  
Finally, install face_recognition  
'pip3 install face_recognition'  
  
For DeepFace, do:  
'pip3 install deepface'  
  
For VGG16 do:  
'pip3 install tensorflow  
pip3 install keras'
