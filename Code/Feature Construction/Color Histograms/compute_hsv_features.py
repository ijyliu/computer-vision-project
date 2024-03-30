
import numpy as np
from PIL import Image
from skimage.color import rgb2hsv

def compute_hsv_features(path: str, bins: int = 256) -> np.ndarray:
    """
    Compute HSV histograms as feature vectors for a given image filepath.

    This function processes the image by converting it to the HSV color space and then
    computes histograms for each of the HSV channels. These histograms provide a statistical
    representation of color distribution in the image.

    Parameters:
    - path (str): The path for the image for which HSV features are to be computed.
    - bins (int, optional): The number of bins to use for the histograms. Default is 256.

    Returns:
    - np.ndarray: A concatenated array of the histograms for Hue, Saturation, and Value channels.
    """

    # Read and convert the image to the HSV color space
    img = np.array(Image.open(path))
    hsv_img = rgb2hsv(img)

    # Compute histograms for each channel
    hist_hue = np.histogram(hsv_img[:, :, 0], bins=bins, range=(0, 1))[0]
    hist_saturation = np.histogram(hsv_img[:, :, 1], bins=bins, range=(0, 1))[0]
    hist_value = np.histogram(hsv_img[:, :, 2], bins=bins, range=(0, 1))[0]

    # Concatenate histograms to form the feature vector
    hsv_features = np.concatenate((hist_hue, hist_saturation, hist_value))

    return hsv_features
