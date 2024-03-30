#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
from PIL import Image
import urllib
from skimage.color import rgb2gray
from skimage.feature import hog


def compute_hog(path: str) -> np.ndarray:
    
    """
    Compute Histogram of Oriented Gradients (HOG) features for all images within a specified directory.

    This function processes each image found in the provided directory path, performing a series of preprocessing steps
    including conversion to grayscale, rescaling to a standard size (256x256 pixels), and normalization of pixel intensity
    values. It then computes the HOG features for each preprocessed image. The HOG feature extraction includes specifying
    the number of orientations, pixels per cell, and cells per block to effectively capture the shape and texture 
    information of the images.

    Parameters:
    - path (str): The directory path containing the images for which HOG features are to be computed.

    Returns:
    - np.ndarray: The number of columns in the array equals the dimensionality of the HOG feature space.

    Note:
    - The function assumes all images are of significant size and can be rescaled to 256x256. For images already at
      this size, the rescaling step is effectively a no-op.
    - The function implicitly handles both grayscale and RGB images, converting RGB images to grayscale as part
      of preprocessing.
    """
    
    # List to hold file names
    impaths = []

    # Loop through directory
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            impaths.append(os.path.join(path, filename))
            
    features = []
    
    for i, impath in enumerate(impaths):
        
        # Read image -------------------------------------------------------------------------
        img = np.array(Image.open(impath))

        # convert to floating point image with intensity [0, 1]
        if np.max(img) > 1:
            img = img.astype(np.float32) / 255.0

        # convert to grayscale
        if len(img.shape) > 2:
            img = rgb2gray(img)
        
        # Rescaling (Optional if image is already 256X256) ------------------------------------
        standard = 256
        scale = standard / min(img.shape[:2])
        img = rescale(img, scale, anti_aliasing = True)
        img = img[int(img.shape[0]/2 - standard/2) : int(img.shape[0]/2 + standard/2),
                           int(img.shape[1]/2 - standard/2) : int(img.shape[1]/2 + standard/2)]
        
        # Compute HOG -------------------------------------------------------------------------
        pixels_per_cell = (16, 16) # Adjusted as per the image size
        cells_per_block = (3, 3) # Adjusted as per the image size
        orientations = 9 # More orientations better for capturing the distinct features of the car

        hog_features = hog(img,
                           orientations = orientations,
                           pixels_per_cell = pixels_per_cell,
                           cells_per_block = cells_per_block)
        features.append(hog_features)
        
    # Convert to an array    
    features = np.array(features)
    return features