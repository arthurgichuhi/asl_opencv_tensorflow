import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
import math
import time
import os

phoneVideo='C:\\Users\\Alkebulan\\Desktop\\PythonProjects\\objectDetection\\signLanguageProject\\cameraVideo.mp4'


cap=cv2.VideoCapture(0)

detector=HandDetector(maxHands=1)

# classifier=Classifier('C:\\Users\\Alkebulan\\Desktop\\PythonProjects\\objectDetection\\signLanguageProject\\model\\keras_model.h5','C:\\Users\\Alkebulan\\Desktop\\PythonProjects\\objectDetection\\signLanguageProject\\model\\labels.txt')

imgSize=300
offset=20

counter=0

labels=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

folder='C:\\Users\\Alkebulan\\Desktop\\PythonProjects\\objectDetection\\signLanguageProject\\Data\\Q'

path='C:\\Users\\Alkebulan\\Desktop\\PythonProjects\\objectDetection\\signLanguageProject\\output\\'

videoName='video.mp4'

outputPath=path+videoName

pre_imgs=os.listdir(path)

for video in pre_imgs:
    if video == videoName:
        videoName=f"${len(pre_imgs)}"+videoName


cv2_fourCC=cv2.VideoWriter_fourcc(*'mp4v')

video=cv2.VideoWriter(outputPath,cv2_fourCC,10,[640,480])
while True:
    success,img=cap.read()
    imgOutput=img.copy()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        
        imgCropShape=imgCrop.shape
        
        imgWhite[0:imgCropShape[0],0:imgCropShape[1]]=imgCrop
        
        aspectRation=h/w
        currentindex=0
        
        if aspectRation>1:
            
            k=imgSize/h
            wCall=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCall,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCall)/2)
            imgWhite[:,wGap:wCall+wGap]=imgResize
            
            # prediction,index=classifier.getPrediction(imgWhite,draw=False)
            # currentindex=index
            # print(prediction,labels[index])
            
        else:
            
            k=imgSize/w
            hCall=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCall))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCall)/2)
            imgWhite[hGap:hCall+hGap,:]=imgResize
            
            # prediction,index=classifier.getPrediction(imgWhite,draw=True)
            # currentindex=index
            
        # cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x+50,y-offset-50+70),(0,0,0),cv2.FILLED)
        # cv2.putText(imgOutput,labels[currentindex],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,1.5,(255,255,255),2)
        
        
        # cv2.imshow('Croped',imgCrop)
        cv2.imshow('Image white',imgWhite)
        
    video.write(imgOutput)
    cv2.imshow('Image',imgOutput)
    # cv2.waitKey(1)
    size=list(imgOutput.shape)
    del size[2]
    
    key=cv2.waitKey(1)
    # if key==ord("q"):
    #     video.release()
    if key== ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}\\{time.time()}.jpg',imgWhite)
        print(counter)