
# Head Pose Estimator

Realtime head pose estimation and renderer to validate results 

## Demo


![Description of the GIF](https://github.com/lovish-sachdev/head-pose-estimator-and-renderer/blob/main/media/gif1.gif?raw=True)


![Description of the GIF](https://github.com/lovish-sachdev/head-pose-estimator-and-renderer/blob/main/media/gif2.gif?raw=True)


![Description of the GIF](https://github.com/lovish-sachdev/head-pose-estimator-and-renderer/blob/main/media/gif3.gif?raw=True)
## Working

The whole process is completed in two major steps:
    
    1. Facial Landmark Detection : for detecting face and extracting 6 reqiured landmarks
    
    2. Rotation angle detection: predicting angles value with these 6 coordinates   


* Mediapipe is used to extract facial landmarks 
* then these landmarks are feed into deel learning model and rotation angle in degrees are obtained
* then a 3d face (iron man mask) model is rotated about these angles 

## Dataset preparation

I made a custom program to prepare synthetic dataset.

* The basic idea is to reverse engineer the problem 
* The position coordinates of 6 points in an arbitarary cordinate system were available 
* I randomly generated x,y,z or roll, pitch ,yaw angles and transformed those 6 points according to angles now these new 12 values (6 cordinates each giving x,y) are our feature vector and x,y,z angles are labels
* then a generator function in python generates batches of data for training  


## data preparation sample code

![App Screenshot](https://github.com/lovish-sachdev/head-pose-estimator-and-renderer/blob/main/media/Screenshot%202024-05-20%20181536.png?raw=true)


## Usage

* clone the repository 

* install libraries from requirements.txt 

* run finall_app.py
## Tech Stack

* vtk :- for rendering 3d model
* mediapipe :- for facial landmark detection
* cv2,pillo,numpy :- for handling images
* tensorflow :- for training model
* streamlit :- for deployment in web

##  References

* head pose using opencv (https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)
