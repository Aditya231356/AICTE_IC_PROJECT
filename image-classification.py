import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Pre-trained model (MobileNetV2 in this case)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to load and preprocess the image
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to fit model input size
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Preprocess the image for MobileNetV2
    return img_array

# Function to predict the image class
def predict_image_class(img_path):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)  # Make prediction
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]  # Decode top 3 predictions
    
    print("Top Predictions:")
    for pred in decoded_predictions:
        print(f"Class: {pred[1]}, Probability: {pred[2]*100:.2f}%")

# Example usage

img_path =input ("enter image path")  # Replace with your image file path
predict_image_class(img_path)