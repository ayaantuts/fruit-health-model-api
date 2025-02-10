GEMINI_API_KEY = "AIzaSyACpDwXspWMdrsqHdM19akpMsg5g0Wkl3A" # Ali bhai ke gemini ka key

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf  # type: ignore
import cv2
import requests
import base64
import tempfile
import PIL.Image
import os
import google.generativeai as genai

app = Flask(__name__)
CORS(app)  # Allow CORS for frontend interaction

# Configure the generative AI library with your API key (if required)
genai.configure(api_key=GEMINI_API_KEY)

# Load your trained TensorFlow model (replace with your actual model file)
tf_model = tf.keras.models.load_model("my_model.keras")

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

def preprocess_image(file_bytes):
    """Preprocess image to match model input requirements."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode the image. Please provide a valid image file.")
    image = cv2.resize(image, (224, 224))  # Resize to model's input shape
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.route('/ingredientsfetch', methods=['POST'])
def ingredientsfetch():
    try:
        # Expect an uploaded file with the key 'image'
        image_file = request.files['image']
        
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_image_path = temp_file.name
            image_file.save(temp_image_path)
        
        # Read the image bytes for base64 encoding
        with open(temp_image_path, "rb") as f:
            file_bytes = f.read()
        encoded_image = base64.b64encode(file_bytes).decode('utf-8')
        
        # Build the prompt for the generative AI model
        prompt = (
            f"Image (base64 encoded): {encoded_image}\n"
            "Extract the information of disease in the above fruit or vegetable image, and provide the reason, symptoms, treatment and whether it is safe to eat."
        )
        
        # Initialize the generative model (use a distinct variable name)
        gen_model = genai.GenerativeModel('gemini-1.5-flash')
        # Adjust the method call as required by the API; here we assume a synchronous call returning an object with a .text attribute.
        result = gen_model.generate_text(prompt=prompt)
        
        # Clean up the temporary image file
        os.remove(temp_image_path)
        
        return jsonify({'result': result.text})
    
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Predict the class of an uploaded image and analyze it using the Gemini API."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    # Read the file once so its content can be used both for prediction and in the prompt
    file_bytes = file.read()

    # Preprocess the image for prediction
    image = preprocess_image(file_bytes)
    predictions = tf_model.predict(image)
    class_id = np.argmax(predictions)
    predicted_class = class_labels[class_id]

    # Encode the image into a base64 string to include in the prompt.
    encoded_image = base64.b64encode(file_bytes).decode('utf-8')
    prompt = (
        f"Image (base64 encoded): {encoded_image}\n"
        f"Predicted Class: {predicted_class}\n"
        "Please provide a JSON response containing the following keys:\n"
        " - 'disease': Describe any disease found in the fruit or vegetable.\n"
        " - 'reason': Explain the reason behind the disease or classification.\n"
        " - 'cure for human': Provide information on what to do if a human has already ingested it.\n"
        " - 'it is safe to eat': Indicate whether it is safe to eat.\n"
        "Ensure the response is valid JSON."
    )

    # Call the Gemini API with the prompt
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    gemini_payload = {
        "prompt": prompt,
        "max_tokens": 150,    # Adjust parameters as needed
        "temperature": 0.5
    }
    # Replace the URL below with the actual Gemini API endpoint.
    gemini_url = "https://api.gemini.example/v1/generate"
    gemini_response = requests.post(gemini_url, headers=headers, json=gemini_payload)

    if gemini_response.status_code != 200:
        return jsonify({
            "error": "Gemini API request failed",
            "details": gemini_response.text
        }), 500

    gemini_result = gemini_response.json()  # Expecting a JSON response from Gemini

    # Construct the final JSON response
    response = {
        "predicted_class": predicted_class,
        "confidence": float(predictions[0][class_id]),
        "gemini_analysis": gemini_result
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)  # Run with debug mode enabled
