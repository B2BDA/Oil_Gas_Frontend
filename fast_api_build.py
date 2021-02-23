import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
from tensorflow import keras
from io import BytesIO
model = keras.models.load_model("ship_classifier_custom.h5")
class_dict = {'1':'Cargo', '2':'Military', '3':'Carrier', '4':'Cruise', '5':'Tankers'}

app = FastAPI()

def read_imagefile(image):
    size = (224, 224)
    image = Image.open(BytesIO(image))
    image = image.resize(size)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    img = np.expand_dims(image_array, axis = 0)
    img = img/255
    return img

def predict(model, img):
    predicted_class = class_dict.get(str(np.argmax(model.predict(img), axis = -1)[0]+1))
    return predicted_class
    
    

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(model, image)
    return prediction

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
