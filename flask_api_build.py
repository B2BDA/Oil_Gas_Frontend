from importlib import reload
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from io import BytesIO
from tensorflow import keras
from PIL import Image
import numpy as np
from pymongo import MongoClient
import base64
client = MongoClient("mongodb+srv://admin:admin@cluster0.42ai5.mongodb.net/test")

try:
    db = client[client.list_database_names()[0]]
    collection = db.list_collection_names()[0]
except Exception as e:
    pass
finally:
    pass

# model = keras.models.load_model("ship_classifier_custom.h5")
model = keras.models.load_model("ship_classifier.h5")
class_dict = {'1':'Cargo', '2':'Military', '3':'Carrier', '4':'Cruise', '5':'Tankers'}
app=Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def read_imagefile(image):
    size = (224, 224)
    image = Image.open(image)
    image = image.resize(size)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    img = np.expand_dims(image_array, axis = 0)
    img = img/255
    return img

def predict(model, img):
    predicted_class = class_dict.get(str(np.argmax(model.predict(img), axis = -1)[0]+1))
    return predicted_class
    

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded')
            image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = read_imagefile(image)
            encoded_string = base64.b64encode(img)
            rec = db[collection].find_one({'img_name':str(encoded_string)})
            if rec is None:
                prediction = predict(model,img)
                print(prediction)
                db[collection].insert_one({'class': str(prediction),'img_name': str(encoded_string)})
                print("Files uploaded successfully")
            else:
                prediction = rec.get('class')
            return render_template('upload.html', result = prediction)
            # return prediction
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)
    


if __name__ == "__main__":
    app.run(host = '127.0.0.1',port = 5000, debug = False, use_reloader=True)