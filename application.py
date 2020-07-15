from flask import Flask,request,jsonify,render_template,send_file
import numpy as np
import cv2
from PIL import Image
import urllib
from datauri import DataURI
import smtplib
import io
import warnings
warnings.filterwarnings("ignore")

def make_data_url(filename):
    prefix = 'data:image/png;base64,'
    fin = open(filename, 'rb')
    #print(type(fin))
    contents = fin.read()
    #print(type(contents))
    import base64
    data_url = prefix + str(base64.b64encode(contents))[2:][:-1]
    fin.close()
    return data_url

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def object_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return(img)


app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/contact', methods=['POST'])
def contact():
    contact_dict = request.values
    name = contact_dict['Name']
    email = contact_dict['Email']
    message = contact_dict['Message']
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login("emailID","password")
    msg = "\r\n".join([
          "From: fromID",
          "To: toID",
          "Subject: Response",
          "",
          name+" "+email+" "+message
          ])
    server.sendmail("fromID","toID", msg)
    return('Done')

@app.route('/hooks', methods=['GET','POST'])
def get_image():
    image_b64 = request.values['imageBase64']
    #print(image_b64[:100])
    response=urllib.request.urlopen(image_b64)
    image=Image.open(response)
    image=image.convert('RGB')
    img=np.asarray(image)
    arr=object_detect(img)
    img = Image.fromarray(arr.astype('uint8'))
    img.save('cache.jpg')
    url=make_data_url('cache.jpg')
    return url

if __name__ == '__main__':
    app.run(debug=True,port="5000",ssl_context='adhoc')
