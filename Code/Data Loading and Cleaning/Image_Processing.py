# Image_Processing.py
# Functions for image processing

# Packages
import cv2
import os
import pandas as pd
from scipy import ndimage

# Function to resize images
# If both dimensions larger than 260 (256 but adding some padding), downsample so the smaller dimension is 260
# Crop the center 256x256 pixels and return image
def resize_image(image):
    # Load width and height
    width = image.shape[1]
    height = image.shape[0]
    # Check if both dimensions are larger than 260
    if width > 260 and height > 260:
        # Determine scaling factor needed to make the smaller dimension 260
        scale_factor = 260 / min(width, height)
        # Add blur to image to handle aliasing
        image = ndimage.gaussian_filter(image, sigma=0.75)
        # Scale image
        image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    # Otherwise: still add some Gaussian blur to handle aliasing (not as optimal as the downsampling case, but still better than nothing)
    else:
        image = ndimage.gaussian_filter(image, sigma=0.75)
    # Crop center 256x256 pixels
    resized_image = image[int((image.shape[0] - 256) / 2):int((image.shape[0] + 256) / 2), int((image.shape[1] - 256) / 2):int((image.shape[1] + 256) / 2)]
    return resized_image

# Function to process an image
def process_image(row):
    # Load image
    image = cv2.imread(os.path.expanduser(row['source_file_path']))
    #print('loaded ', row['source_file_path'])
    # Resize image
    resized_image = resize_image(image)
    # Save image
    cv2.imwrite(os.path.expanduser(row['destination_file_path']), resized_image)
