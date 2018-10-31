import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


video = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =video.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf>50):
            if(Id==1):
                Id="Vedhas "+str(conf)
            elif(Id==2):
                Id="Also Vedhas "+str(conf)
        else:
            Id="Unknown "+str(conf)
        cv2.putText(im,str(Id),(x,y+h),font,1,(0,255,0))
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==27:
        break
video.release()
cv2.destroyAllWindows()
