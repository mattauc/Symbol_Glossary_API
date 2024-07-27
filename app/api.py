from flask import Flask, request, jsonify
from .dataset import transforms, classes
from .model import OCRNet
from PIL import Image
import io
import torch
from app import app

print("api run")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = OCRNet(len(classes)).to(device)

# Load the trained model weights
#model.load_state_dict(torch.load('path_to_your_model_weights.pth', map_location=device))

# Set the model to evaluation mode
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale

    # Apply the transform
    img = transforms(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img)  # Forward pass through the model
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted.item()]

    return jsonify({'class': predicted_class})
