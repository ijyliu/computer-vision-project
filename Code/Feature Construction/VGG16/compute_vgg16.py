import numpy as np
#from tensorflow.keras.preprocessing.image import img_to_array, load_img
#from tensorflow.keras.applications import VGG16
#from tensorflow.keras.applications.vgg16 import preprocess_input
import torch
#import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

# Preprocess the image to convert to a suitable format for VGG16
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet's mean and std
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image
    
def compute_vgg16(model, device, image_path):
    """Compute VGG16 features"""
    
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path).to(device)

    # Extract features
    with torch.no_grad():
        features = model(image)

        # Add a pooling layer to convert features from (1, 512, 7, 7) to (1, 512, 1, 1)
        features = nn.AdaptiveAvgPool2d((1, 1))(features)
        features = features.view(features.size(0), -1)
    
    return features
