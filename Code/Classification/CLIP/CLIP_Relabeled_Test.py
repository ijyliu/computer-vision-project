# CLIP Relabeled Test
# Check accuracy of off-the-shelf CLIP classifier on relabeled test data set, unambiguous items.
# Source: https://auto.gluon.ai/stable/tutorials/multimodal/image_prediction/clip_zeroshot.html

# Packages
from autogluon.multimodal import MultiModalPredictor
from IPython.display import Image, display
import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# Flag for a test/sample run
test_run = False

# Function to take an input filepath and return the CLIP probabilities
def get_clip_prediction(filepath):

    # Load image from filepath
    loaded_img = Image(filename=filepath)

    # Get CLIP prediction
    predictor = MultiModalPredictor(problem_type="zero_shot_image_classification")
    probs = predictor.predict_proba({"image": [loaded_img]}, {"text": ['This is an SUV', 'This is a Sedan', 'This is a Pickup', 'This is a Convertible']})

    # Return probabilities
    return probs

def main():

    # Load "relabeled_test_no_blur_old_and_new_labels.xlsx"
    relabeled_test_no_blur_old_and_new_labels = pd.read_excel('../../../Data/Relabeled_Test_No_Blur/relabeled_test_no_blur_old_and_new_labels.xlsx')

    # For testing, limit to 10 rows
    if test_run:
        relabeled_test_no_blur_old_and_new_labels = relabeled_test_no_blur_old_and_new_labels.head(10)

    # Print the first few rows of the dataframe
    print(relabeled_test_no_blur_old_and_new_labels.head())

    # Delete if 'New Class' is not 'SUV', 'Sedan', 'Pickup', or 'Convertible'
    relabeled_test_no_blur_old_and_new_labels = relabeled_test_no_blur_old_and_new_labels[relabeled_test_no_blur_old_and_new_labels['New Class'].isin(['SUV', 'Sedan', 'Pickup', 'Convertible'])]

    # Construct image path
    relabeled_test_no_blur_old_and_new_labels['non_blurred_image_path'] = '../../../Images/test/No Blur/' + relabeled_test_no_blur_old_and_new_labels['filename']

    # Determine the number of available CPUs
    num_cpus = os.cpu_count()
    print('number of CPUs:', num_cpus)

    # Use a process pool to execute image processing in parallel
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit the image processing function to the executor using map
        probs = list(executor.map(get_clip_prediction, relabeled_test_no_blur_old_and_new_labels['non_blurred_image_path']))

    # Check first value of probs
    print(probs[0])
    
    # Add probabilities to the dataframe
    relabeled_test_no_blur_old_and_new_labels['clip_probs'] = probs
    # Unnest the probabilities lists
    relabeled_test_no_blur_old_and_new_labels['prob_SUV'] = relabeled_test_no_blur_old_and_new_labels['clip_probs'].apply(lambda x: x[0])
    relabeled_test_no_blur_old_and_new_labels['prob_Sedan'] = relabeled_test_no_blur_old_and_new_labels['clip_probs'].apply(lambda x: x[1])
    relabeled_test_no_blur_old_and_new_labels['prob_Pickup'] = relabeled_test_no_blur_old_and_new_labels['clip_probs'].apply(lambda x: x[2])
    relabeled_test_no_blur_old_and_new_labels['prob_Convertible'] = relabeled_test_no_blur_old_and_new_labels['clip_probs'].apply(lambda x: x[3])

    # Check dataframe
    print(relabeled_test_no_blur_old_and_new_labels.head())

    # Get predicted label
    relabeled_test_no_blur_old_and_new_labels['predicted_label'] = relabeled_test_no_blur_old_and_new_labels[['prob_SUV', 'prob_Sedan', 'prob_Pickup', 'prob_Convertible']].idxmax(axis=1)
    # Map to label
    relabeled_test_no_blur_old_and_new_labels['predicted_label'] = relabeled_test_no_blur_old_and_new_labels['predicted_label'].map({'prob_SUV': 'SUV', 'prob_Sedan': 'Sedan', 'prob_Pickup': 'Pickup', 'prob_Convertible': 'Convertible'})

    # Check dataframe
    print(relabeled_test_no_blur_old_and_new_labels.head())

    # Check accuracy
    accuracy = (relabeled_test_no_blur_old_and_new_labels['predicted_label'] == relabeled_test_no_blur_old_and_new_labels['New Class']).mean()
    print('Accuracy:', accuracy)

    # Output predictions to Excel
    # Keep columns filename, prob_SUV, prob_Sedan, prob_Pickup, prob_Convertible, predicted_label, 'New Class'
    relabeled_test_no_blur_old_and_new_labels[['filename', 'prob_SUV', 'prob_Sedan', 'prob_Pickup', 'prob_Convertible', 'predicted_label', 'New Class']].to_excel('../../Data/Predictions/CLIP_Relabeled_Test_No_Blur/CLIP_Relabeled_Test_No_Blur_predictions.xlsx', index=False)

if __name__ == "__main__":
    main()
