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

    # HOG 16 ppc features
    hog16_feature_columns = [col for col in all_features_train.columns if 'HOG_16_ppc' in col]
    hog16_features = np.array(all_features_train[hog16_feature_columns])
    print('hog16 features shape:', hog16_features.shape)

    # HOG 24 ppc features
    hog24_feature_columns = [col for col in all_features_train.columns if 'HOG_24_ppc' in col]
    hog24_features = np.array(all_features_train[hog24_feature_columns])
    print('hog24 features shape:', hog24_features.shape)

    # HSV features
    hsv_feature_columns = [col for col in all_features_train.columns if 'HSV' in col]
    hsv_features = np.array(all_features_train[hsv_feature_columns])
    print('hsv features shape:', hsv_features.shape)

    # Vision transformer features
    vit_feature_columns = [col for col in all_features_train.columns if 'ViT' in col]
    vit_features = np.array(all_features_train[vit_feature_columns])
    print('vit features shape:', vit_features.shape)

    # All features
    all_features_df = all_features_train.drop(columns=['Class', 'harmonized_filename', 'test_80_20'])
    print('items in all_features_df columns not in hog16, hog24, hsv, vit:', [col for col in all_features_df.columns if col not in hog16_feature_columns + hog24_feature_columns + hsv_feature_columns + vit_feature_columns])
    all_features = np.array(all_features_df)
    print('all features shape:', all_features.shape)
    print('sum of smaller features matrix widths: ', hog16_features.shape[1] + hog24_features.shape[1] + hsv_features.shape[1] + vit_features.shape[1])

    # List of feature matrices (can be used as X_list in later functions)
    feature_matrices = [hog16_features, hog24_features, hsv_features, vit_features, all_features]
    # Rescale feature matrices
    feature_matrices_rescaled = [StandardScaler().fit_transform(feature_matrix) for feature_matrix in feature_matrices]
    # Associated labels
    feature_matrix_labels = ['HOG 16 PPC Features', 'HOG 24 PPC Features', 'HSV Features', 'Vision Transformer Features', 'All Features']

    # Return constructed items
    return y, class_mapping, feature_matrices, feature_matrices_rescaled, feature_matrix_labels
