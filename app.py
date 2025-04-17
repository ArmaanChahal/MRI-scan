from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__, static_folder='static', template_folder='templates')

model = tf.keras.models.load_model('mri_model.keras')
class_names = ['merged-glioma', 'merged-meningioma', 'merged-no_tumor', 'merged-pituitary']
IMAGE_SIZE = 128

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'File is empty'}), 400

    try:
        image_bytes = file.read()
        image_array = preprocess_image(image_bytes)
        prediction = model.predict(image_array)
        predicted_index = int(np.argmax(prediction[0]))
        predicted_class = class_names[predicted_index].replace("merged-", "").replace("_", " ").title()
        confidence = round(float(prediction[0][predicted_index]) * 100, 2)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
