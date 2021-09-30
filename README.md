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
