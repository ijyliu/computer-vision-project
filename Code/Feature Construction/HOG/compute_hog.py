import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

# Function to compute HOG Features
def compute_hog(path: str) -> np.ndarray:
    
    """
    Compute Histogram of Oriented Gradients (HOG) feature for a given image filepath.

    This function processes the image found, performing a series of preprocessing steps
    including conversion to grayscale and normalization of pixel intensity
    values. It then computes the HOG features. The HOG feature extraction includes specifying
    the number of orientations, pixels per cell, and cells per block to effectively capture the shape and texture 
    information of the image.

    Parameters:
    - path (str): The path for the image for which HOG features are to be computed.

    Returns:
    - np.ndarray: The size of the array equals the dimensionality of the HOG feature space.

    Note:
    - The function implicitly handles both grayscale and RGB images, converting RGB images to grayscale as part
      of preprocessing.
    """
        
    # Read image -------------------------------------------------------------------------
    img = np.array(Image.open(path))

    # convert to floating point image with intensity [0, 1]
    if np.max(img) > 1:
        img = img.astype(np.float32) / 255.0

    # convert to grayscale
    if len(img.shape) > 2:
        img = rgb2gray(img)

    # Compute HOG -------------------------------------------------------------------------
    pixels_per_cell = (24, 24) # Adjusted as per the image size
    cells_per_block = (3, 3) # Adjusted as per the image size
    orientations = 4 # More orientations better for capturing the distinct features of the car

    hog_features = hog(img,
                       orientations = orientations,
                       pixels_per_cell = pixels_per_cell,
                       cells_per_block = cells_per_block)
        
    return hog_features
