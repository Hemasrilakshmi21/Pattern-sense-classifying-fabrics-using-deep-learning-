from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# Load pre-trained model
model = load_model('model/fabric_model.h5')  # Make sure the path is correct
class_names = ['Striped', 'Floral', 'Checkered', 'Polka Dot', 'Geometric']
# Home route
@app.route('/')
def home():
    return render_template('index.html')
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."
    file = request.files['file']
    if file.filename == '':
        return "No selected file."
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))  # change size based on your model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize if model trained on normalized input
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        return render_template('index.html', prediction=predicted_class, image_url=filepath)
    return "Something went wrong."
if __name__ == '__main__':
    app.run(debug=True)
    


