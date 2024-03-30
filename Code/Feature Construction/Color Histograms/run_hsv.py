import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor  
from compute_hsv_features import compute_hsv_features
import time

def load_image_paths(base_path, blur_option, sample_run=False, sample_size=2):
    """
    Load image paths from the specified directory, with an option for a sample run.

    Parameters:
    - base_path (str): Base directory path for the images.
    - blur_option (str): Subdirectory name to choose between blurred or non-blurred images.
    - sample_run (bool, optional): Flag to indicate a sample run. Default is False.
    - sample_size (int, optional): Number of samples to include in a sample run. Default is 2.

    Returns:
    - list: A list of full image paths.
    """
    image_paths = [os.path.join(base_path, blur_option, img) for img in os.listdir(os.path.join(base_path, blur_option))]
    return image_paths[:sample_size] if sample_run else image_paths

def compute_features(image_paths):
    """
    Compute HSV features for a list of image paths using a ThreadPoolExecutor.

    Parameters:
    - image_paths (list): A list of image paths for which to compute features.

    Returns:
    - list: A list of HSV feature vectors.
    """
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor: 
        feature_vectors = list(executor.map(compute_hsv_features, image_paths))
    return feature_vectors

def main():
    start_time = time.time()


    sample_run = True  
    blur_no_blur = 'Blurred' 

    base_path = '../../../Images'  
    train_image_paths = load_image_paths(os.path.join(base_path, 'train'), blur_no_blur, sample_run)
    test_image_paths = load_image_paths(os.path.join(base_path, 'test'), blur_no_blur, sample_run)

    #compute features
    all_image_paths = train_image_paths + test_image_paths
    feature_vectors = compute_features(all_image_paths)

    #Prepare DataFrame
    feature_df = pd.DataFrame(feature_vectors, columns=[f'hsv_feature_{i}' for i in range(len(feature_vectors[0]))])
    feature_df['Image Path'] = all_image_paths
    feature_df['test_80_20'] = [0] * len(train_image_paths) + [1] * len(test_image_paths)

    # Save DataFrame
    feature_df.to_csv('../../../Data/HSV/hsv_features.csv', index=False)

    print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
