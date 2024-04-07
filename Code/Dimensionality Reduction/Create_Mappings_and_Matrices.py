# Create Mappings and Matrices

# Packages
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function for loading features from parquet files
def combine_directory_parquets(directory_path):
    '''
    Combines all parquet files in a directory into a single dataframe.
    '''
    # If path does not end in a slash, add one
    if directory_path[-1] != '/':
        directory_path += '/'
    # list of files in directory
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.parquet')]
    # read in all parquet files
    combined_df = pd.concat([pd.read_parquet(directory_path + f) for f in file_list])
    # Return combined dataframe
    return combined_df

# Function for creating mappings and matrices
def create_mappings_and_matrices():
    '''
    Creates y, class mapping, and the feature matrices and labels. Returns y, class mapping, list of feature matrices, list of rescaled feature matrices, and list of labels. 
    Update this function as more features are added.
    '''

    # All Features - train data
    all_features_train = combine_directory_parquets('../../Data/Features/All Features/train')
    print('all features train')
    print(all_features_train)

    # Create column y
    # Encode Class as 0 for SUV, 1 for Sedan, 2 for Pickup, 3 for Convertible
    class_mapping = {'SUV': 0, 'Sedan': 1, 'Pickup': 2, 'Convertible': 3}
    y = all_features_train['Class'].map(class_mapping)
    print('first 5 y values:', y[:5])

    # Create Feature Matrices - Update This Code As More Features Are Added
    feature_shorthand = ['HOG_16_ppc', 'HOG_24_ppc', 'HSV', 'LBP', 'VGG', 'ViT']
    # Associated labels
    feature_matrix_labels = ['HOG 16 PPC Features', 
                             'HOG 24 PPC Features', 
                             'HSV Features', 
                             'LBP Features',
                             'VGG Features',
                             'Vision Transformer Features']
    # List of feature columns for each feature group
    feature_columns_lists = []
    # List of feature matrices (can be used as X_list in later functions)
    feature_matrices = []
    for feature_folder_name in feature_shorthand:
        feature_columns = [col for col in all_features_train.columns if feature_folder_name in col]
        feature_columns_lists.append(feature_columns)
        feature_matrices.append(np.array(all_features_train[feature_columns]))
        print(feature_folder_name, 'features shape:', feature_matrices[-1].shape)

    # All features
    # Just keep numeric columns
    all_features_df = all_features_train.select_dtypes(include='number')
    print('items in all_features_df columns not in feature groups (hog, vgg, etc.):', [col for col in all_features_df.columns if col not in [col for sublist in feature_columns_lists for col in sublist]])
    all_features = np.array(all_features_df)
    print('all features shape:', all_features.shape)
    print('sum of smaller features matrix widths: ', sum([feature_array.shape[1] for feature_array in feature_matrices]))
    # Add all features to feature matrices
    feature_matrices.append(all_features)
    # Add label
    feature_matrix_labels.append('All Features')

    # Rescale feature matrices
    feature_matrices_rescaled = [StandardScaler().fit_transform(feature_matrix) for feature_matrix in feature_matrices]

    # Return constructed items
    return y, class_mapping, feature_matrices, feature_matrices_rescaled, feature_matrix_labels
