import cv2,os
import numpy as np
from PIL import Image

video=cv2.VideoCapture(0)
classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id=input('enter id: ')
sample=0

while(True):
    ret,img=video.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite("faces/user."+id+'.'+str(sample)+".jpg",gray[y:y+h,x:x+w])
        sample=sample+1
        cv2.imshow('frame',img)
    if cv2.waitKey(100) & 0xFF == 27:
        break
    elif sample>50:
        break
video.release()
cv2.destroyAllWindows()

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        if(os.path.split(imagePath)[-1].split(".")[-1]!='jpg'):
            continue
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids


faces,Ids = getImagesAndLabels('faces')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainer/trainer.yml')
