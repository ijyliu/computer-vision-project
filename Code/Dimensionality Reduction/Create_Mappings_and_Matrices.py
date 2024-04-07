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
    Creates y, class mapping, and the feature matrices and labels for both train and test. Returns y_train, y_test, class mapping, list of feature matrices for train, list of feature matrices for test, list of rescaled feature matrices for train, list of rescaled feature matrices for test, and list of labels. 
    Update this function as more features are added.
    '''

    # All Features - train data
    all_features_train = combine_directory_parquets('../../Data/Features/All Features/train')
    print('all features train')
    print(all_features_train)

    # All Features - test data
    all_features_test = combine_directory_parquets('../../Data/Features/All Features/test')
    print('all features test')
    print(all_features_test)

    # Create column y
    # Encode Class as 0 for SUV, 1 for Sedan, 2 for Pickup, 3 for Convertible
    class_mapping = {'SUV': 0, 'Sedan': 1, 'Pickup': 2, 'Convertible': 3}
    y_train = all_features_train['Class'].map(class_mapping)
    print('first 5 y values:', y_train[:5])
    y_test = all_features_test['Class'].map(class_mapping)

    # Create Feature Matrices - Update This Code As More Features Are Added
    feature_shorthand = [#'HOG_16_ppc', # (eliminated for excessive dimensionality)
                         'HOG_24_ppc', 
                         'HSV', 
                         'LBP', 
                         'VGG', 
                         'ViT']
    # Associated labels
    feature_matrix_labels = [#'HOG 16 PPC Features', # (eliminated for excessive dimensionality)
                             'HOG 24 PPC Features', 
                             'HSV Features', 
                             'LBP Features',
                             'VGG Features',
                             'Vision Transformer Features']
    # List of feature columns for each feature group
    feature_columns_lists_train = []
    feature_columns_lists_test = []
    # List of feature matrices (can be used as X_list in later functions)
    feature_matrices_train = []
    feature_matrices_test = []
    for feature_folder_name in feature_shorthand:
        feature_columns = [col for col in all_features_train.columns if feature_folder_name in col]
        # Train set
        feature_columns_lists_train.append(feature_columns)
        feature_matrices_train.append(np.array(all_features_train[feature_columns]))
        print(feature_folder_name, 'features shape:', feature_matrices_train[-1].shape)
        # Test set
        feature_columns_lists_test.append([col for col in all_features_test.columns if feature_folder_name in col])
        feature_matrices_test.append(np.array(all_features_test[feature_columns]))
        print(feature_folder_name, 'features shape:', feature_matrices_test[-1].shape)

    # All features
    # Just keep numeric columns

    # Train set
    all_features_df_train = all_features_train.select_dtypes(include='number')
    print('items in all_features_df_train columns not in feature groups (hog, vgg, etc.):', [col for col in all_features_df_train.columns if col not in [col for sublist in feature_columns_lists_train for col in sublist]])
    all_features_train = np.array(all_features_df_train)
    print('all features shape:', all_features_train.shape)
    print('sum of smaller features matrix widths: ', sum([feature_array.shape[1] for feature_array in feature_matrices_train]))
    # Add all features to feature matrices
    feature_matrices_train.append(all_features_train)

    # Test set
    all_features_df_test = all_features_test.select_dtypes(include='number')
    print('items in all_features_df_test columns not in feature groups (hog, vgg, etc.):', [col for col in all_features_df_test.columns if col not in [col for sublist in feature_columns_lists_test for col in sublist]])
    all_features_test = np.array(all_features_df_test)
    print('all features shape:', all_features_test.shape)
    print('sum of smaller features matrix widths: ', sum([feature_array.shape[1] for feature_array in feature_matrices_test]))
    # Add all features to feature matrices
    feature_matrices_test.append(all_features_test)

    # Add label
    feature_matrix_labels.append('All Features')

    # Rescale feature matrices
    feature_matrices_rescaled_train = [StandardScaler().fit_transform(feature_matrix) for feature_matrix in feature_matrices_train]
    feature_matrices_rescaled_test = [StandardScaler().fit_transform(feature_matrix) for feature_matrix in feature_matrices_test]

    # Return constructed items
    return y_train, y_test, class_mapping, feature_matrices_train, feature_matrices_test, feature_matrices_rescaled_train, feature_matrices_rescaled_test, feature_matrix_labels
