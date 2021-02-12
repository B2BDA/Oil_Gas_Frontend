from tensorflow import keras
import numpy as np
# from tensorflow.keras.preprocessing.image import load_img
from PIL import Image, ImageOps
import streamlit as st
model = keras.models.load_model("ship_classifier.h5")
st.title("Ship Classification")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
class_dict = {'1':'Cargo', '2':'Military', '3':'Carrier', '4':'Cruise', '5':'Tankers'}
 
 
print(uploaded_file)
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

    
    if st.button("Predict"):
        predict = np.argmax(model.predict(img), axis = -1)
        predict = class_dict.get(str(np.argmax(model.predict(img), axis = -1)[0]+1))
        st.write("The ship is a : %s ship"%str(predict))
        # make a predicted_class variable and append in the above statement