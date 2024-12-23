# Packages
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from compute_lbp import *
import shutil
import time

# Flag for a test/sample run
sample_run = False
# Feature name
feature_name = 'LBP'
# Blurred or No Blur images
blur_no_blur = 'Blurred'

# Enabling and disabling print
# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__

def split_df(df, dataset_name, out_folder, num_pieces):
    '''
    Splits dataframes into num_pieces and saves them as parquet files in out_folder. Reduces file size to comply with GitHub limits.
    '''
    # Tracking total length of pieces
    total_len_pieces = 0
    # Create out_folder if it does not exist
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # Delete previous pieces, all contents of out_folder
    for filename in os.listdir(out_folder):
        file_path = os.path.join(out_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    # Save pieces
    for i in range(num_pieces):
        # start index for piece rows
        start_index = i * len(df) // num_pieces
        # end index for piece rows
        end_index = (i + 1) * len(df) // num_pieces
        # get piece
        piece = df[start_index:end_index]
        piece.to_parquet(out_folder + '/' + dataset_name + '_piece_' + str(i) + '.parquet', index=False)
        #print(len(piece))
        total_len_pieces += len(piece)
    
    # check total piece length and length of vit_embeddings_df
    print('length check passed')
    print(total_len_pieces == len(df))

def main():

    # Start timer
    start_time = time.time()

    ####################################################################################################

    # Determine the number of available CPUs
    num_cpus = os.cpu_count()
    print('number of CPUs:', num_cpus)

    ####################################################################################################

    # Load train Image paths
    train_image_paths = ['../../../Images/train/' + blur_no_blur + '/' + img_path for img_path in os.listdir('../../../Images/train/' + blur_no_blur + '/')]
    if sample_run:
        train_image_paths = train_image_paths[:2]
    print('first train image path')
    print(train_image_paths[0])
    # Load test Image paths
    test_image_paths = ['../../../Images/test/' + blur_no_blur + '/' + img_path for img_path in os.listdir('../../../Images/test/' + blur_no_blur + '/')]
    if sample_run:
        test_image_paths = test_image_paths[:2]
    print('first test image path')
    print(test_image_paths[0])
    # Concatenate train and test image paths
    paths = train_image_paths + test_image_paths
    # Create dataframe for feature vectors
    feature_vectors_df = pd.DataFrame(paths, columns=['Image Path'])
    # Create variable test_80_20 to indicate if the image is in the test set
    # Repeat 0 for the number of train images and 1 for the number of test images
    feature_vectors_df['test_80_20'] = [0] * len(train_image_paths) + [1] * len(test_image_paths)
    print('value counts of test_80_20')
    print(feature_vectors_df['test_80_20'].value_counts())

    ####################################################################################################

    feature_name = 'LBP'
    variant_df = feature_vectors_df.copy()
    # Settings
    radius = 3
    n_points = 24
    # Convert to lists by repeating the same value for the number of images
    radius_list = [radius] * len(paths)
    n_points_list = [n_points] * len(paths)
    print('Radii')
    print(radius_list[:5])
    print(len(radius_list))

    # Use a process pool to execute image processing in parallel
    # Turn off printing
    blockPrint()
    # Set up pool
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit the image processing function to the executor using map
        feature_vectors = list(executor.map(compute_lbp, variant_df['Image Path'], radius_list, n_points_list))
    # Enable printing
    enablePrint()

    # Check first, second value of feature_vectors
    print('first and second feature vector')
    print(feature_vectors[0])
    print(feature_vectors[1])
    print('length of first feature vector')
    print(len(feature_vectors[0]))

    # Unnest each numpy array in feature_vectors into dataframe columns for variant_df
    # Turn off printing
    blockPrint()
    for i in range(len(feature_vectors[0])):
        variant_df[feature_name + '_' + str(i)] = [vector[i] for vector in feature_vectors]
    # Enable printing
    enablePrint()
    # Check dataframe
    print(variant_df.head())

    # Split and output dataframe
    split_df(variant_df, feature_name, '../../../Data/Features/' + feature_name, 10)

    
    ####################################################################################################

    # End timer
    end_time = time.time()

    # Print time taken in minutes
    ttm = ((end_time - start_time) / 60)
    print('Time taken (in minutes):', ttm)

    # Time per image
    print('Time per image (in minutes):', ttm / len(feature_vectors_df))

if __name__ == "__main__":
    main()
