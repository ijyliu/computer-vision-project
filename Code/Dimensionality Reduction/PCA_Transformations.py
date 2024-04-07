# PCA Transformations

###################################################################################################

# Packages
from Create_Mappings_and_Matrices import *
import pandas as pd
from sklearn.decomposition import PCA
import os
import shutil

###################################################################################################

# Set up number of components to keep for each item feature_matrix_labels here, based on the plot results
num_components_to_keep = {
    'HOG 16 PPC': 1000,
    'HOG 24 PPC': 400,
    'HSV': 400,
    'LBP': 10,
    'VGG': 400,
    'Vision Transformer': 400,
    'All Features': 1500
}

###################################################################################################

# Load data

# Train set
all_features_train = combine_directory_parquets('../../Data/Features/All Features/train')
print('all features train')
print(all_features_train)

# Test set
all_features_test = combine_directory_parquets('../../Data/Features/All Features/test')
print('all features test')
print(all_features_test)

###################################################################################################

# Get mappings and matrices
y_train, y_test, class_mapping, _, _, feature_matrices_rescaled_train, feature_matrices_rescaled_test, feature_matrix_labels = create_mappings_and_matrices()

# Check that y and 'Class' column line up. Class column transformed with class_mapping should be equal to y
# Both items as numpy arrays
# Check for train set
class_col_numpy = all_features_train['Class'].map(class_mapping).to_numpy()
print('y_train and class column numpy arrays are equal:', np.array_equal(y_train.to_numpy(), class_col_numpy))
# Check for test set
class_col_numpy = all_features_test['Class'].map(class_mapping).to_numpy()
print('y_test and class column numpy arrays are equal:', np.array_equal(y_test.to_numpy(), class_col_numpy))

###################################################################################################

# Get transformed matrices
transformed_dfs_train = []
transformed_dfs_test = []
# Iterate through feature_matrix_labels
for label in feature_matrix_labels:
    print('operating on label ', label)
    # Get number of components to keep
    n_components = num_components_to_keep[label]
    print('number of components to keep:', n_components)
    # Get rescaled features
    rescaled_train = feature_matrices_rescaled_train[np.argwhere(np.array(feature_matrix_labels) == label)[0][0]]
    rescaled_test = feature_matrices_rescaled_test[np.argwhere(np.array(feature_matrix_labels) == label)[0][0]]
    # Run PCA, fitting on train
    pca = PCA(n_components=n_components)
    fitted_pca = pca.fit(rescaled_train)
    # Transform train and test
    X_train_pca = fitted_pca.transform(rescaled_train)
    X_test_pca = fitted_pca.transform(rescaled_test)
    # Put in dataframe
    pca_df_train = pd.DataFrame(X_train_pca)
    pca_df_test = pd.DataFrame(X_test_pca)
    # Column names - 'All_Features_PCA_0', 'All_Features_PCA_1', etc.
    pca_df_train.columns = [label + '_PCA_' + str(i) for i in range(n_components)]
    pca_df_test.columns = [label + '_PCA_' + str(i) for i in range(n_components)]
    # Append to transformed_matrices
    transformed_dfs_train.append(pca_df_train)
    transformed_dfs_test.append(pca_df_test)

# Concatenate all dataframes in transformed_dfs_train and transformed_dfs_test that do not correspond to 'All Features'
transformed_df_train_individual_features = pd.concat([df for df in transformed_dfs_train if 'All Features' not in df.columns[0]], axis=1)
transformed_df_test_individual_features = pd.concat([df for df in transformed_dfs_test if 'All Features' not in df.columns[0]], axis=1)

# Get 'All Features' dataframes
transformed_df_train_all_features = [df for df in transformed_dfs_train if 'All Features' in df.columns[0]][0]
transformed_df_test_all_features = [df for df in transformed_dfs_test if 'All Features' in df.columns[0]][0]

# Add on string columns from all_features_train and all_features_test
all_features_train_string_cols = [col for col in all_features_train.columns if all_features_train[col].dtype == 'object']
all_features_test_string_cols = [col for col in all_features_test.columns if all_features_test[col].dtype == 'object']
transformed_df_train_individual_features = pd.concat([all_features_train[all_features_train_string_cols], transformed_df_train_individual_features], axis=1)
print('transformed_df_train_individual_features')
print(transformed_df_train_individual_features)
transformed_df_test_individual_features = pd.concat([all_features_test[all_features_test_string_cols], transformed_df_test_individual_features], axis=1)
print('transformed_df_test_individual_features')
print(transformed_df_test_individual_features)
transformed_df_train_all_features = pd.concat([all_features_train[all_features_train_string_cols], transformed_df_train_all_features], axis=1)
print('transformed_df_train_all_features')
print(transformed_df_train_all_features)
transformed_df_test_all_features = pd.concat([all_features_test[all_features_test_string_cols], transformed_df_test_all_features], axis=1)
print('transformed_df_test_all_features')
print(transformed_df_test_all_features)

# Create save directories
if not os.path.exists('../../Data/Features/All Features All Features PCA/train'):
    os.makedirs('../../Data/Features/All Features All Features PCA/train')
if not os.path.exists('../../Data/Features/All Features All Features PCA/test'):
    os.makedirs('../../Data/Features/All Features All Features PCA/test')
if not os.path.exists('../../Data/Features/All Features Individual Features PCA/train'):
    os.makedirs('../../Data/Features/All Features Individual Features PCA/train')
if not os.path.exists('../../Data/Features/All Features Individual Features PCA/test'):
    os.makedirs('../../Data/Features/All Features Individual Features PCA/test')

###################################################################################################

# Save transformed dataframes

def split_df(df, dataset_name, out_folder, num_pieces):
    '''
    Splits dataframes into num_pieces and saves them as parquet files in out_folder. Reduces file size to comply with GitHub limits.
    '''
    # Tracking total length of pieces
    total_len_pieces = 0
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

# Split dataframes into 16 pieces and save
split_df(transformed_df_train_individual_features, 'train', '../../Data/Features/All Features Individual Features PCA/train', 16)
split_df(transformed_df_test_individual_features, 'test', '../../Data/Features/All Features Individual Features PCA/test', 16)
split_df(transformed_df_train_all_features, 'train', '../../Data/Features/All Features All Features PCA/train', 16)
split_df(transformed_df_test_all_features, 'test', '../../Data/Features/All Features All Features PCA/test', 16)
