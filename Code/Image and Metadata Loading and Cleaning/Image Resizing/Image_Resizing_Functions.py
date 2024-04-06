# Image_Resizing_Functions.py

# Packages
import cv2
import os
from scipy import ndimage
import matplotlib.pyplot as plt # using mpl image read/write to try to avoid potential cv2 issues

def resize_image(image, blur = True, blur_sigma = 0.75, target_size = 256):
    '''
    Function to resize images. If both dimensions larger than target size + 4 (adding some padding), downsample so the smaller dimension is target size + 4. Apply Gaussian blurs to handle aliasing. Crop the center pixels to fit target size square and return image.
    '''
    # Load width and height
    width = image.shape[1]
    height = image.shape[0]
    # Check if both dimensions are larger than target size + 4
    if width > target_size + 4 and height > target_size + 4:
        # Determine scaling factor needed to make the smaller dimension target size + 4
        scale_factor = (target_size + 4) / min(width, height)
        # Optionally blur to image to handle aliasing
        if blur:
            image = ndimage.gaussian_filter(image, sigma=blur_sigma)
        # Scale image
        image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        # Crop center pixels
        resized_image = image[int((image.shape[0] - target_size) / 2):int((image.shape[0] + target_size) / 2), int((image.shape[1] - target_size) / 2):int((image.shape[1] + target_size) / 2)]
    # Otherwise: can still add some Gaussian blur to handle aliasing (not as optimal as the downsampling case, but still better than nothing)
    else:
        # Optionally blur to image to handle aliasing
        if blur:
            image = ndimage.gaussian_filter(image, sigma=blur_sigma)
        # Crop center pixels
        resized_image = image[int((image.shape[0] - target_size) / 2):int((image.shape[0] + target_size) / 2), int((image.shape[1] - target_size) / 2):int((image.shape[1] + target_size) / 2)]
    return resized_image

def image_resizing_read_write(source_file_path, destination_file_path, blur = True, blur_sigma = 0.75, target_size = 256):
    '''
    Function to process an image. Takes the source and destination file paths and resizes appropriately.
    '''
    # Load image
    input_image = plt.imread(os.path.expanduser(source_file_path))
    # Resize image
    resized_image = resize_image(input_image, blur, blur_sigma, target_size)
    # Save image
    plt.imsave(os.path.expanduser(destination_file_path), resized_image)
