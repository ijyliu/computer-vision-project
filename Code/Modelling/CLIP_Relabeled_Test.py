# CLIP Relabeled Test
# Check accuracy of off-the-shelf CLIP classifier on relabeled test data set, unambiguous items.
# Source: https://auto.gluon.ai/stable/tutorials/multimodal/image_prediction/clip_zeroshot.html

# Packages
from autogluon.multimodal import MultiModalPredictor
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# Function to take an input filepath and return the CLIP probabilities
def get_clip_prediction(filepath):

    # Load image from filepath
    plt.imread(filepath)

    # Get CLIP prediction
    predictor = MultiModalPredictor(problem_type="zero_shot_image_classification")
    probs = predictor.predict({"image": [filepath]}, {"text": ['This is an SUV', 'This is a Sedan', 'This is a Pickup', 'This is a Convertible']})

    # Return probabilities
    return probs

def main():

    # Flag for a test/sample run
    test_run = False

    # Determine the number of available CPUs
    num_cpus = os.cpu_count()
    print('number of CPUs:', num_cpus)

    # Use a process pool to execute image processing in parallel
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit the image processing function to the executor
        for img_filepath in img_filepaths:
            executor.submit(get_clip_prediction, img_filepath)

if __name__ == "__main__":
    main()
