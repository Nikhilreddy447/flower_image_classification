from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow import keras

app = Flask(__name__)

# Load the saved model
loaded_model = keras.models.load_model('my_model.h5')

# Map the predicted class index to the corresponding flower label
class_labels = {0: 'roses', 1: 'daisy', 2: 'dandelion', 3: 'sunflowers', 4: 'tulips'}

# Define the function to preprocess and predict on a single image
def predict_image(image):
    resized_image = cv2.resize(image, (180, 180))
    input_image = np.expand_dims(resized_image, axis=0) / 255.0  # Normalize the image
    prediction = loaded_model.predict(input_image)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    return predicted_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'imageInput' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['imageInput']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        prediction = predict_image(img)
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
