import cv2
import numpy as np
from time import sleep
from keras.preprocessing.image import img_to_array,ImageDataGenerator
from keras.models import load_model

face_classifier = cv2.CascadeClassifier("/home/sky/Documents/2.CNN/Emotion and Facial Expression Detector/haarcascade/haarcascade_frontalface_default.xml")

classifier = load_model("/home/sky/Documents/h5_model/emo_det.h5")


class_labels = {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Neutral', 4: 'Sad', 5: 'Surprise'}
classes = list(class_labels.values())


def face_detector(img):
    #convert img to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    
    
    for (x,y,w,h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        
    
    try:
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((48,48), np.uint8), img
    return  (x,w,y,h), roi_gray, img


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    
    if np.sum([face]) != 0.0:
        roi = face.astype("float32") /255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        
        #make a prediction on the ROI then lookup the class
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        label_position = (rect[0] +int((rect[1]/2)), rect[2]+25)
        
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    else:
        cv2.putText(image, "No face Found", (20,60), cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
        
    
    cv2.imshow('All', image)
    if cv2.waitKey(1) == 13:  #13  is Enter key
        break

cap.release()
cv2.destroyAllWindows()