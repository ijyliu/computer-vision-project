import cv2
from skimage.feature import local_binary_pattern
import numpy as np

def compute_lbp(path: str, radius=1, n_points=8) -> np.ndarray:
    """
    Compute Local Binary Pattern (LBP) feature histogram for a given image filepath.

    This function processes the image found, performing a series of preprocessing steps
    including conversion to grayscale. It then computes the LBP features and returns
    the histogram of the LBP features.

    Parameters:
    - path (str): The path for the image for which LBP features are to be computed.
    - radius (int): Radius of circle (spatial resolution of the operator).
    - n_points (int): Number of points sampled by the LBP operator.

    Returns:
    - np.ndarray: The computed LBP feature histogram.

    Note:
    - The function handles both grayscale and RGB images, converting RGB images to grayscale as part
      of preprocessing.
    """
    # Read image
    img = cv2.imread(path)
    
    # Check if image loading is successful
    if img is None:
        raise ValueError(f"Error: Unable to read image at path '{path}'")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute LBP
    lbp_image = local_binary_pattern(gray, n_points, radius, method='uniform')

    # Compute histogram of LBP image
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize

    return hist

# Example usage:
folder_path = '../../../Images/single class samples/Blur'
try:
    lbp_histogram = compute_lbp(folder_path)
    print("LBP Histogram:", lbp_histogram)
except ValueError as e:
    print(e)
