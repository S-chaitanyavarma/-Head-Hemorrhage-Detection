from flask import Flask, render_template, request, send_from_directory
import cv2

from tensorflow.keras.models import Sequential,load_model
import numpy as np
import pickle
import pandas as pd
model1 = load_model('keras_model.h5',compile=True)
labels_dict={0:'problem',1:'normal'}
COUNT = 0




app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')\

@app.route('/project_details')
def details():
    return render_template('details.html')


@app.route('/home', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224,224,3)#(1, 224,224,3)
    result = model1.predict(img_arr)
    
    label=np.argmax(result,axis=1)[0]
    prediction = labels_dict[label]
   
    if prediction=="problem":
             print("problem")
    elif prediction== "normal": 
            print("normal")
  
          
    COUNT +=1;
    return render_template('prediction.html', data=prediction)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))



if __name__ == '__main__':
    app.run(debug=True)



