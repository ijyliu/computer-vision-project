# app.py
# Web app to accept uploaded image and return the predicted class

##################################################################################################

# Packages
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import re
import base64
from io import BytesIO
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

##################################################################################################

# Model code

# Standard ResNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load model
# Initialize ResNet50
model = models.resnet50(pretrained=True)
# Modify the final layer for our classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
# Load state
model.load_state_dict(torch.load('resnet.pth', map_location=torch.device('cpu')))
# Put model in evaluation mode
model.eval()

# Load label mapping
label_mapping = pd.read_excel('label_mapping.xlsx')
# Convert to dictionary
label_mapping = label_mapping.to_dict()['Class']

def get_image_prediction(img, model, processor):
    '''
    Function to get the prediction for an image.
    
    Parameters:
    - img: PIL image
    - model: PyTorch model
    - processor: PyTorch preprocessor

    Returns:
    - img_class_str: Predicted class as string
    '''
    # Convert image to RGB
    image = img.convert("RGB")
    # Pre-process image
    preprocessed_image = processor(image)
    # Run the image through the model
    # Add batch dimension
    with torch.no_grad():
        outputs = model(torch.unsqueeze(preprocessed_image, 0))
    # Get class as number
    img_class = torch.argmax(outputs).item()
    # Use mapping to get class as string
    img_class_str = label_mapping[img_class]
    return img_class_str

##################################################################################################

# App code

# Declare a flask app
app = Flask(__name__)
CORS(app)

# Main page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Utility function to convert base64 image data to PIL image
def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image

# Prediction endpoint
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        # Make prediction
        result = get_image_prediction(img, model, preprocess)
        # Serialize the result
        return jsonify(result=result)
    return None

# Serve the app with gevent
if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()
