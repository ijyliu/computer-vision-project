import os
from concurrent.futures import ProcessPoolExecutor
from Image_Resizing_Functions import *
import pandas as pd
#import shutil

# Flag for a test/sample run
test_run = True

def main():

    # Load "resized_cars_annos.xlsx"
    # This file contains the paths to the original images and destinations to save to
    resized_cars_annos = pd.read_excel('../../../Data/resized_cars_annos.xlsx')

    # Column for "test" or "train" based on test_80_20
    resized_cars_annos['test_train'] = resized_cars_annos['test_80_20'].apply(lambda x: 'test' if x == 1 else 'train')

    # Construct Blurred and Non-Blurred Image Paths
    resized_cars_annos['blurred_image_path'] = '../../../Images/' + resized_cars_annos['test_train'] + '/Blurred/' + resized_cars_annos['blurred_destination_file_name']
    resized_cars_annos['non_blurred_image_path'] = '../../../Images/' + resized_cars_annos['test_train'] + '/No Blur/' + resized_cars_annos['no_blur_destination_file_name']

    # Blurred images
    # Create a list of tuples containing the source and destination paths for the images
    # Columns orig_res_file_path and resized_file_path
    blurred_images = list(zip(resized_cars_annos['orig_res_file_path'], resized_cars_annos['blurred_image_path']))
    # If this is a test run, only do the first 10 images
    if test_run:
        blurred_images = blurred_images[:10]
    print('number of blurred images to process:', len(blurred_images))
    # Determine the number of available CPUs
    num_cpus = os.cpu_count()
    print('number of CPUs:', num_cpus)
    # Use a process pool to execute image processing in parallel
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit the image processing function to the executor
        for source_path, destination_path in blurred_images:
            executor.submit(image_resizing_read_write, source_path, destination_path, blur=True, blur_sigma=0.75, target_size=256)

    # Non-Blurred images
    # Create a list of tuples containing the source and destination paths for the images
    # Columns orig_res_file_path and resized_file_path
    non_blurred_images = list(zip(resized_cars_annos['orig_res_file_path'], resized_cars_annos['non_blurred_image_path']))
    # If this is a test run, only do the first 10 images
    if test_run:
        non_blurred_images = non_blurred_images[:10]
    print('number of non-blurred images to process:', len(non_blurred_images))
    # Determine the number of available CPUs
    num_cpus = os.cpu_count()
    print('number of CPUs:', num_cpus)
    # Use a process pool to execute image processing in parallel
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit the image processing function to the executor
        for source_path, destination_path in non_blurred_images:
            executor.submit(image_resizing_read_write, source_path, destination_path, blur=False, target_size=256)

if __name__ == "__main__":
    main()
