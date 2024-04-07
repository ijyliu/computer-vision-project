import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from compute_hsv_features import compute_hsv_features
import shutil
import time

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def split_df(df, dataset_name, out_folder, num_pieces):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for filename in os.listdir(out_folder):
        file_path = os.path.join(out_folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    total_len_pieces = 0
    for i in range(num_pieces):
        start_index = i * len(df) // num_pieces
        end_index = (i + 1) * len(df) // num_pieces
        piece = df.iloc[start_index:end_index]
        piece.to_parquet(os.path.join(out_folder, f'{dataset_name}_piece_{i}.parquet'))
        total_len_pieces += len(piece)

    assert total_len_pieces == len(df), "Length check failed after splitting."

def main():
    start_time = time.time()

    sample_run = True
    blur_no_blur = 'Blurred'
    num_pieces = 10

    train_image_paths = [os.path.join('../../../Images/train', blur_no_blur, img) for img in os.listdir(os.path.join('../../../Images/train', blur_no_blur))]
    test_image_paths = [os.path.join('../../../Images/test', blur_no_blur, img) for img in os.listdir(os.path.join('../../../Images/test', blur_no_blur))]

    if sample_run:
        train_image_paths = train_image_paths[:2]
        test_image_paths = test_image_paths[:2]

    image_paths = train_image_paths + test_image_paths
    test_80_20_labels = [0] * len(train_image_paths) + [1] * len(test_image_paths)

    blockPrint()
    with ProcessPoolExecutor() as executor:
        feature_vectors = list(executor.map(compute_hsv_features, image_paths))
    enablePrint()

    feature_df = pd.DataFrame(feature_vectors)
    feature_df['Image Path'] = image_paths
    feature_df['test_80_20'] = test_80_20_labels

    split_df(feature_df, 'hsv_feature', '../../../Data/HSV', num_pieces)

    print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()