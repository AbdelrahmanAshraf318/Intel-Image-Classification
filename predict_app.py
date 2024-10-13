import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
    global model
    model = tf.keras.models.load_model('saved_model/network.h5')
    print(" * Model Loaded! ")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model.....")
get_model()

@app.route('/predict', methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(100,100))

    prediction = model.predict(processed_image).tolist()

    #Classes are:
    #'Buildings','Forest','Glacier','Mountain','Sea','Street'

    response = {
        'prediction': {
            'Buildings': prediction[0][0],
            'Forest': prediction[0][1],
            'Glacier': prediction[0][2],
            'Mountain': prediction[0][3],
            'Sea': prediction[0][4],
            'Street': prediction[0][5]
        }
    }
    return jsonify(response)
