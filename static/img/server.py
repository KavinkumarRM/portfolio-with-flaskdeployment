import flask
import numpy
import numpy as np
import cv2
from PIL import Image
from keras.models import model_from_json
import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout,Activation,Flatten
from keras.optimizers import Adam
#from keras import regularizers
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
import joblib
#from sklearn.decomposition import PCA
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
from scipy import misc
import io
from flask import Flask, request, redirect, url_for, jsonify, Response, render_template
from flask_cors import CORS, cross_origin
import json
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")

def object_detection_api(img):
    learning_rate = .001

    faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    #loading the facenet embedding architecture
    i=0
    json_file = open('./model_resampling_1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./facenet_resampling_1.h5")
    loaded_model.compile(loss='mean_absolute_error', optimizer=Adam())

    #loading the lower classifier
    classifier=joblib.load('./GBM_resampling_1.joblib.pkl')

    labelss=["deni", "sriram", "rohit", "bhadra", "kavin"]

    def get_faces(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )
        return(faces)

    def get_predictions(img):
        faces=get_faces(img)
        img_height, img_width, channels = img.shape
        output=[]
        for (x,y,w,h) in faces:
            framess=np.asarray(img[y-int(0.1*h):y+h+int(0.1*h),x-int(0.1*w):x+w+int(0.1*w)])
            framess=cv2.resize(framess,(150,150))
            framess=np.divide(framess,255)
            score=loaded_model.predict(framess.reshape(1,150,150,3))
            cls=classifier.predict(score)
            text=labelss[cls[0]]
            temp_dict={}
            temp_dict["name"]="Object"
            temp_dict['class_name']=text
            temp_dict['height']=float(h/img_height)
            temp_dict['width']=float(w/img_width)
            temp_dict['score']=1
            temp_dict['y']=float(y/img_height)
            temp_dict['x']=float(x/img_width)
            output.append(temp_dict)
        return(json.dumps(temp_dict))
    return(get_predictions(img))


app = Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response

@app.route('/image', methods=['POST'])
def image():
    if request.method == 'POST':
        try:
            image_file = request.files['image']  # get the image

            # Set an image confidence threshold value to limit returned data
            """threshold = None #request.form.get('threshold')
            if threshold is None:
                threshold = 0.5
            else:
                threshold = float(threshold)"""

            # finally run the image through tensor flow object detection`
            image_object = Image.open(image_file)
            objects = object_detection_api.get_objects(image_object, threshold)
            """objects=json.dumps([{"threshold": 0.5, "name": "webrtcHacks Sample Tensor Flow Object API Service 0.0.1", "numObjects": 10},
            {"name": "Object", "class_name": "person", "height": 0.5931246876716614, "width": 0.40913766622543335, "score": 0.916878342628479, "y": 0.5538768172264099, "x": 0.39422380924224854},
            {"name": "Object", "class_name": "kite", "height": 0.40220093727111816, "width": 0.3590298891067505, "score": 0.8294452428817749, "y": 0.3829464316368103, "x": 0.34582412242889404}])"""
            return objects

        except Exception as e:
            print('POST /image error: %e' % e)
            return e

@app.route('/local',methods=['GET'])
def local():
    return render_template('local.html')

if __name__=='__main__':
    app.run(port = 5000 , debug=True)
