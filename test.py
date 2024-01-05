import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
loaded_model = keras.models.load_model('my_model.h5')

# Define the function to preprocess and predict on a single image
def predict_image(image_path, model):
    img = cv2.imread(image_path)
    resized_image = cv2.resize(img, (180, 180))
    input_image = np.expand_dims(resized_image, axis=0) / 255.0  # Normalize the image
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Provide the path to the input image
input_image_path = '10683189_bd6e371b97.jpg'

# Make predictions
predicted_class = predict_image(input_image_path, loaded_model)

# Map the predicted class index to the corresponding flower label
class_labels = {0: 'roses', 1: 'daisy', 2: 'dandelion', 3: 'sunflowers', 4: 'tulips'}
predicted_label = class_labels[predicted_class]

print(f'The predicted class is: {predicted_label}')