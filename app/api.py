from flask import Flask, request, jsonify
from .dataset import EVAL_TRANSFORMS, classes
from .model import OCRNet
from PIL import Image
import io
import torch
import webbrowser
from torchvision.transforms import ToPILImage
from app import app
import os


to_pil = ToPILImage()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = OCRNet(len(classes)).to(device)

# Load the trained model weights
model.load_state_dict(torch.load('model_weights.pth', map_location=device))

# Set the model to evaluation mode
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale

    try:
        img_transformed = EVAL_TRANSFORMS(img)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Convert tensor back to PIL image for saving
    img_pil = to_pil(img_transformed.cpu().squeeze(0))  # Remove batch dimension and convert to PIL
    file_path = "transformed_image.png"
    img_pil.save(file_path)

    # Automatically open the saved image
    webbrowser.open(os.path.abspath(file_path))
    # Apply the transform
    img = EVAL_TRANSFORMS(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img)  # Forward pass through the model
        #print(output)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted.item()]
        print(predicted_class)

    return jsonify({'symbol': predicted_class})
