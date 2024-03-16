import os
from concurrent.futures import ProcessPoolExecutor
from Image_Resizing_Functions import *
import pandas as pd
import shutil

# Flag for a test/sample run
test_run = False

def main():

    # Load "~/Box/INFO 290T Project/Intermediate Data/resized_cars_annos.xlsx"
    # This file contains the paths to the original and resized images
    resized_cars_annos = pd.read_excel('~/Box/INFO 290T Project/Intermediate Data/resized_cars_annos.xlsx')

    # Create a list of tuples containing the source and destination paths for the images
    # Columns orig_res_file_path and resized_file_path
    images = list(zip(resized_cars_annos['orig_res_file_path'], resized_cars_annos['resized_file_path']))

    # If this is a test run, only do the first 10 images
    if test_run:
        images = images[:10]

    print('number of images to process:', len(images))

    # Prepare directories
    # Clear existing directory
    shutil.rmtree(os.path.expanduser("~/Box/INFO 290T Project/Intermediate Data/Resized Images/"), ignore_errors=True)
    # Make subdirectories
    os.makedirs(os.path.expanduser("~/Box/INFO 290T Project/Intermediate Data/Resized Images/train"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/Box/INFO 290T Project/Intermediate Data/Resized Images/test"), exist_ok=True)
    
    # Determine the number of available CPUs
    num_cpus = os.cpu_count()

    print('number of CPUs:', num_cpus)

    # Use a process pool to execute image processing in parallel
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit the image processing function to the executor
        for source_path, destination_path in images:
            executor.submit(process_image, source_path, destination_path)
            # print('files in target dir')
            # print(len(os.listdir(os.path.expanduser(destination_path) + '/..')))

if __name__ == "__main__":
    main()
