from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf # type: ignore
import cv2

app = Flask(__name__)
CORS(app)  # Allow CORS for frontend interaction

# Load your trained model (replace 'model.h5' with your actual model file)
# model = tf.keras.models.load_model("fruit_vegetable_disease_detection_model.h5")
model = tf.keras.models.load_model("my_model.keras")


# Class labels (replace with your actual class names)
class_labels = [
 'Apple__Healthy',
 'Apple__Rotten',
 'Banana__Healthy',
 'Banana__Rotten',
 'Bellpepper__Healthy',
 'Bellpepper__Rotten',
 'Carrot__Healthy',
 'Carrot__Rotten',
 'Cucumber__Healthy',
 'Cucumber__Rotten',
 'Grape__Healthy',
 'Grape__Rotten',
 'Guava__Healthy',
 'Guava__Rotten',
 'Jujube__Healthy',
 'Jujube__Rotten',
 'Mango__Healthy',
 'Mango__Rotten',
 'Orange__Healthy',
 'Orange__Rotten',
 'Pomegranate__Healthy',
 'Pomegranate__Rotten',
 'Potato__Healthy',
 'Potato__Rotten',
 'Strawberry__Healthy',
 'Strawberry__Rotten',
 'Tomato__Healthy',
 'Tomato__Rotten'
 ]

def preprocess_image(image):
    """Preprocess image to match model input requirements."""
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Resize to model's input shape
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.route("/predict", methods=["POST"])
def predict():
    """Predict the class of an uploaded image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    image = preprocess_image(file)
    predictions = model.predict(image)
    class_id = np.argmax(predictions)
    print(predictions)
    predicted_class = class_labels[class_id]
    
    return jsonify({"predicted_class": predicted_class, "confidence": float(predictions[0][class_id])})

if __name__ == "__main__":
    app.run(debug=True)  # Run with debug mode enabled
