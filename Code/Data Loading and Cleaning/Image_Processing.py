# Image_Processing.py
# Functions for image processing

# Packages
import cv2
import os
from scipy import ndimage
import matplotlib.pyplot as plt # using mpl image read/write to try to avoid potential cv2 issues
import time

def resize_image(image):
    '''
    Function to resize images. If both dimensions larger than 260 (256 but adding some padding), downsample so the smaller dimension is 260. Apply Gaussian blurs to hand aliasing. Crop the center 256x256 pixels and return image.
    '''
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

def process_image(source_file_path, destination_file_path):
    '''
    Function to process an image. Takes the source and destination file paths and resizes appropriately.
    '''
    # Load image
    image = plt.imread(os.path.expanduser(source_file_path))
    # Check image is loaded
    if image is None:
        raise ValueError('Image not loaded: ' + os.path.expanduser(source_file_path))
    # Resize image
    resized_image = resize_image(image)
    # Check image is resized
    if resized_image is None:
        raise ValueError('Image not resized: ' + os.path.expanduser(source_file_path))
    # Save image
    plt.imsave(os.path.expanduser(destination_file_path), resized_image)
    # Check file exists
    # Sleep for a little bit to avoid potential file system issues
    time.sleep(0.1)
    if not os.path.exists(os.path.expanduser(destination_file_path)):
        raise ValueError('Image not saved: ' + os.path.expanduser(destination_file_path))
