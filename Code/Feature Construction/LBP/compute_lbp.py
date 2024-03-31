import cv2
from skimage.feature import local_binary_pattern
import numpy as np

def compute_lbp(path: str, radius=1, n_points=8) -> np.ndarray:
    """
    Compute Local Binary Pattern (LBP) feature for a given image filepath.

    This function processes the image found, performing a series of preprocessing steps
    including conversion to grayscale. It then computes the LBP features.

    Parameters:
    - path (str): The path for the image for which LBP features are to be computed.
    - radius (int): Radius of circle (spatial resolution of the operator).
    - n_points (int): Number of points sampled by the LBP operator.

    Returns:
    - np.ndarray: The computed LBP feature vector.

    Note:
    - The function handles both grayscale and RGB images, converting RGB images to grayscale as part
      of preprocessing.
    """
    # Read image
    img = cv2.imread(path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform').flatten()

    return lbp
