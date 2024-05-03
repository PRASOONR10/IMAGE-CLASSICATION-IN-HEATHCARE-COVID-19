import os
import numpy as np

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with tf.keras model.save()
MODEL_PATH = 'models/my_model.h5'

# Load your trained model
model = load_model(MODEL_PATH, compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        if preds[0][0] >=  0.5:
            prediction = 'Positive For Covid-19'
        else:
            prediction = 'Negative for Covid-19'              
        return prediction
    return None


if __name__ == '__main__':
    app.run(debug=True)
