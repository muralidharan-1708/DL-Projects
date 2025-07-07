from flask import Flask, request, render_template
#request --> interface between html and app
#render_template --> 
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)

model = load_model('model.hdf5')

def prepare_image(image):
    image = ImageOps.grayscale(image)  #gray_scaling
    image = ImageOps.invert(image)  #convert black to white or vise versa to read the images easily
    image = image.resize((28, 28)) #covert image to our model input shape
    image = np.array(image)/255  #normalization
    image = image.reshape(1, 28, 28, 1)   #reshape(batchsize, height, weight, channel)
    return image

@app.route('/', methods = ['GET','POST']) #to communicate with client - server
def index():
    if request.method == 'POST':  
        file = request.files['file']
        if file:
            image = Image.open(file.stream)  #file stream --> read the file without saving
            image = prepare_image(image)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            return render_template('result.html', predicted_class = predicted_class)
    return render_template('index.html')

if(__name__ == '__main__'):
    app.run(debug = True, use_reloader = False) #debug - tell if the error occurs in the runtime. use_reloader --> avoid running flask two times