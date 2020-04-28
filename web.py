from flask import Flask, render_template,request
import numpy as np
from PIL import Image
import PIL
import cv2
from copy import copy
import os
import pickle
import time

os.chdir('C:\my files\ece-hackathon')
app = Flask(__name__)


framesss = []


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit',methods=['POST'])
def submit():
    n = request.form.get('name')
    a = request.form.get('age')
    m = request.form.get('phone')
    folder = 'images/'+n
    os.mkdir(folder)
    file_loc = folder+'/x.jpg'
    i=0





    face_cas = cv2.CascadeClassifier('C:\my files\ece-hackathon\cassFiles\haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    os.chdir('C:\my files\ece-hackathon')
    while(i<10):
        ret,frame = cap.read()
        cpy_frame = copy(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0),2)
            x2=int((x+w))
            y2=int((y+h)*2/3)
            roi_gray = gray[y:y2, x:x2]
            roi_color = frame[y:y2, x:x2]

        cv2.imshow('vdo', frame)

        i+=1
        time.sleep(0.5)
        framesss.append(cpy_frame)

    c=1
    for im in framesss:
        name = "images/"+n+'/'+str(c)+'.jpg'
        cv2.imwrite(name,cpy_frame)
        c+=1

    cap.release()
    cv2.destroyAllWindows()
    train()


    return "successful"





def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "images")

    face_cas = cv2.CascadeClassifier('C:\my files\ece-hackathon\cassFiles\haarcascade_frontalface_default.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    os.chdir('C:\my files\ece-hackathon')
    x_train = []
    y_lables = []
    current_id = 0
    lable_ids = {}
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                lable = os.path.basename(root).replace(" ", "-").lower()
                if not lable in lable_ids:
                    lable_ids[lable]= current_id
                    current_id +=1
                id_ = lable_ids[lable]
                print(lable_ids)


                pil_image = Image.open(path).convert("L")
                image_array = np.array(pil_image, "uint8")
                #print(image_array)
                faces = face_cas.detectMultiScale(image_array,1.5,5)
                for (x,y,w,h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_lables.append(id_)

    with open("lables.pickle", 'wb') as f:
        pickle.dump(lable_ids, f)

    recognizer.train(x_train, np.array(y_lables))
    recognizer.save("traied.yml")
