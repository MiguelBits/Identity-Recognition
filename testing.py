import cv2
import numpy as np
import pickle
import statistics
from PIL import Image
import os
import shutil
from shutil import copyfile
from pathlib2 import Path
from random import randint

question = raw_input("Do you want to train before testing? [y/n] ")

def train_faces():
    print(" running train_faces.py ... ")
    BASE_DIR = os.path.dirname(os.path.abspath("C:\Users\mcruzvas\Desktop\ID\src\images"))
    image_dir = os.path.join(BASE_DIR, "images")

    face_cascade = cv2.CascadeClassifier("C:\Users\mcruzvas\Desktop\ID\src\haarcascade_frontalface_alt2.xml")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    x_train = []
    y_labels = []
    current_id = 1000
    label_ids = {}

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1             
                id_ = label_ids[label]
                print(label_ids)
                
                #y_labels.append(label) #some number value for our labels
                #x_train.append(path)# verify this image, turn into NUMPY ARRAY, GRAY
                
                pil_image = Image.open(path).convert("L") #grayscale
                size = (550,550)
                final_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(final_image, "uint8")
                #print(image_array)
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=3)

                for (x,y,w,h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)
    #print(y_labels)
    #print(x_train)

    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")
    print(" \nTrainer is now trained :) \n")
    print(" \nnow will run main.py... \n")

if question == "y" :
    train_faces()

if question == "n":
    print(" running main.py... ")
    
face_cascade = cv2.CascadeClassifier("C:\Users\mcruzvas\Desktop\ID\src\haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("C:\Users\mcruzvas\Desktop\ID\src\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("C:\Users\mcruzvas\Desktop\ID\src\haarcascade_smile.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={"Leonel Messi": 1000,
        "Miguel de Vasconcelos": 1001,
        "Mila Kunis": 1002,
        "Miley Cyrus": 1003,
        "Obama Barack": 1004,
        "Nikolaj Coster-Waldau": 1005
        }
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}
    
cap = cv2.VideoCapture(0)


def hello():
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x,y,w,h) in faces:
            #print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #region of interest
            roi_color = frame[y:y+h, x:x+w]

            id_, conf = recognizer.predict(roi_gray) #recognizer deep learn model
            modeid_ = []
            modeid_.append(id_)
            maxconf = []
            maxconf.append(conf)

            conf_person = {}  #conf associated with person
            conf_person[str(labels[id_])] = str(conf)  #person : conf
            while len(maxconf) < 51:
                
                #print(conf," confianca ", labels[id_])
                maxconf.append(conf)
            maxmax = max(maxconf) # max of maxconf
            #print(max(maxconf))   
            while len(modeid_) < 51:
                modeid_.append(id_)
                #print(modeid_," lista")
                #print("statistics.mode: ", statistics.mode(modeid_))
            Mid_ = statistics.mode(modeid_) #mode of id_
            if conf >= 50 and conf <= 99 :
                #print(id_)
                #print(labels[id_])
                cv2.putText(frame, labels[Mid_], (x,y), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255),2, cv2.LINE_AA)
            #cv2.putText(frame, labels[Mid_], (x,y), cv2.FONT_ITALIC,1,(0,255,0),2,cv2.LINE_AA)
            cv2.imwrite("myIMG.png", roi_color) #last img taken by webcam

            cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0), 2)
        #           widht = x+w height = y+h , color = green , thickness = 2

            strmaxmax = str(maxmax)
            #print(conf_person[strmaxmax], labels[Mid_])
            #if conf_person[labels[Mid_]] == str(maxmax):
                #Hello = ("Hello %s :)" %(labels[Mid_]))
            #hellolist = []
            #hellolist.append(Hello)
            #print(hellolist)
            #if hellolist[-1] != Hello:
                #print(Hello,"\nBe welcome!")

            '''
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)
            smile = smile_cascade.detectMultiScale(roi_gray)
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh),(0,255,0),2)'''
            
        cv2.imshow('webcam', frame)
        if conf_person[labels[Mid_]] == str(maxmax):
            Hello = ("Hello %s :)" %(labels[Mid_]))
            break
        if cv2.waitKey(1) == ord("q"):
            break
    print(Hello,"\nBe Welcome!")
    #print(labels[Mid_])
    strMid_ = ''.join(labels[Mid_])
    #print(strMid_.lower())
    filenumber = randint(0,999)
    my_file = Path("C:\Users\mcruzvas\Desktop\ID\src\images\%s\myIMG%s.png"%(strMid_.lower(), filenumber))

    if my_file.is_file():
        copyfile("C:\Users\mcruzvas\Desktop\ID\myIMG.png", "C:\Users\mcruzvas\Desktop\ID\src\images\%s\myIMG%s.png"%(strMid_.lower(), randint(0,999)))
    else:
        copyfile("C:\Users\mcruzvas\Desktop\ID\myIMG.png", "C:\Users\mcruzvas\Desktop\ID\src\images\%s\myIMG%s.png"%(strMid_.lower(), filenumber))
        
    cap.release()
    cv2.destroyAllWindows()

hello()
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml'); #front face cv2 frontal
cam = cv2.VideoCapture(0);  #0 for 1st webcam detected

while(True):
    ret,img=cam.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #Rectangle on face         BGR value,255green , 2 de thickness
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')): #1ms delay, break se keyvalue = q
        break;
cam.release()
cv2.destroyAllWindows()