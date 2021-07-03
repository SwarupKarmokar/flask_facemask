from flask import Flask, render_template, Response
import cv2
import pickle
import numpy as np
# import winsound
from pandas.core import frame
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

with_mask = np.load('with_mask1.npy')
without_mask = np.load('without_mask.npy')

with_mask = with_mask.reshape(400, 28*28*3)
without_mask = without_mask.reshape(400, 28*28*3)

X = np.r_[with_mask, without_mask]
y = np.zeros(X.shape[0])
y[400:] = 1.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
X_train = pca.fit_transform(X_train)

my_model = pickle.load(open('facemasksvc.pkl','rb'))

names = {0: 'Mask', 1 : 'No_Mask'}

haar = cv2.CascadeClassifier('data.xml')
def face_img(img):
    cords = haar.detectMultiScale(img)
    return cords

cam = cv2.VideoCapture(0)
def gen_frame():
    while True:
        font = cv2.FONT_HERSHEY_COMPLEX
        ret, frame = cam.read()

        cord = face_img(frame)
        for x,y,w,h in cord:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,200,255), 4)
        
            face = frame[y:y+h, x:x+w, :]
            face = cv2.resize(face, (28,28))

            face = face.reshape(1,-1)
            face = pca.transform(face)
            pred = my_model.predict(face)
            output = names[int(pred)]

           
            if output == "No_Mask":
                cv2.putText(frame, output, (x,y), font, 1, (0,0,255), 2)              
#                 winsound.Beep(frequency=1000, duration=100)
                
            elif output == "Mask":
                cv2.putText(frame, output, (x,y), font, 1, (0,255,0), 2)

        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()

          
            

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
