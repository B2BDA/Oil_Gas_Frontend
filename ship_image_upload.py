from tensorflow import keras
import numpy as np
# from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import streamlit as st
from pymongo import MongoClient
client = MongoClient("mongodb+srv://admin:admin@cluster0.42ai5.mongodb.net/test")

model1 = keras.models.load_model("ship_classifier.h5")
model2 = keras.models.load_model("ship_classifier_vgg.h5")
st.title("Ship Classification")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
class_dict = {'1':'Cargo', '2':'Military', '3':'Carrier', '4':'Cruise', '5':'Tankers'}
try:
    db = client[client.list_database_names()[0]]
    collection = db.list_collection_names()[0]
except Exception as e:
    pass
finally:
    pass

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # this image goes into the predict function of the model
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
 
    # Create the array of the right shape to feed into the keras model
    # data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    #image sizing
    size = (224, 224)
    image = image.resize(size)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    img = np.expand_dims(image_array, axis = 0)
    img = img/255

    
    # if st.button("Predict"):
    option = st.radio( 'Choose Model?',('MobileNetV2', 'VGG16', 'ResNet'))
    rec = db[collection].find_one({'img_name':str(uploaded_file.name)})
    if option == 'MobileNetV2':
        if rec is None:
            # predict = np.argmax(model1.predict(img), axis = -1)
            predict = class_dict.get(str(np.argmax(model1.predict(img), axis = -1)[0]+1))
            st.write("The ship is a : %s ship"%str(predict))
            # make a predicted_class variable and append in the above statement
            db[collection].insert_one({'class': str(predict),'img_name': str(uploaded_file.name)})
        else:
            rec = rec.get('class')
            st.write("The ship is a : %s ship"%str(rec))
    elif option == "VGG16":
        if rec is None:
            predict = np.argmax(model2.predict(img), axis = -1)
            predict = class_dict.get(str(np.argmax(model2.predict(img), axis = -1)[0]+1))
            st.write("The ship is a : %s ship"%str(predict))
            # make a predicted_class variable and append in the above statement
            db[collection].insert_one({'class': str(predict),'img_name': str(uploaded_file.name)})
        else:
            rec = rec.get('class')
            st.write("The ship is a : %s ship"%str(rec))
    else:
        st.write("Model coming soon")