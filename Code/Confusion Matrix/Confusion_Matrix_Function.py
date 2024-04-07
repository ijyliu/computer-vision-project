import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
from PIL import Image
import numpy as np
import random

def analyze_classifier_performance(base_path, classifier_name,images_root_dir):
    
    classifier_colors = {
        'Logistic Regression': 'Blues',
        'XGBoost': 'Reds',
        'SVM': 'Greens'
    }
    
    classifier_dir = os.path.join(base_path, classifier_name)
    file_pattern = os.path.join(classifier_dir, f'{classifier_name.replace(" ", "_")}*.xlsx')
    files = glob.glob(file_pattern)

    if not files:
        print(f"No Excel files found for classifier '{classifier_name}' in '{classifier_dir}'.")
        return

    file_path = files[0]
    data = pd.read_excel(file_path)
    actual_labels = data['Class']
    predicted_column = [col for col in data.columns if 'Classification' in col][0]
    predicted_labels = data[predicted_column]

    #Classification Report
    report = classification_report(actual_labels, predicted_labels, output_dict=True)
    pd.DataFrame(report).transpose().to_excel(f'{classifier_name}_Classification_Report.xlsx')

     # Confusion Matrix
    cm_color = classifier_colors.get(classifier_name, 'Blues')
    cm = confusion_matrix(actual_labels, predicted_labels, labels=actual_labels.unique())
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cm_color, xticklabels=actual_labels.unique(), yticklabels=actual_labels.unique())
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title(f'Confusion Matrix for {classifier_name}')
    plt.savefig(f'{classifier_name}_Confusion_Matrix.png')
    plt.close()

    # Misclassified Images
    misclassified_indices = np.where(actual_labels != predicted_labels)[0]
    selected_indices = random.sample(list(misclassified_indices), min(5, len(misclassified_indices)))
    if selected_indices:
        plt.figure(figsize=(20, 4))
        for i, index in enumerate(selected_indices, 1):
            excel_image_path = data.iloc[index]['image_path_no_blur'].strip()
            adjusted_image_path = excel_image_path.lstrip('.').lstrip('/').split('/', 3)[-1]
            full_image_path = os.path.join(images_root_dir, adjusted_image_path)
            image = Image.open(full_image_path)
            plt.subplot(1, 5, i)
            plt.imshow(image)
            plt.title(f'{actual_labels[index]} Misclassified As {predicted_labels[index]}')
            plt.axis('off')
        plt.savefig(f'{classifier_name}_Misclassified_Images.png')
        plt.close()
#Example usage of the function:
# visualize_classifier_confusion_matrix('path/to/Predictions', 'Logistic Regression','../../Images')
# visualize_classifier_confusion_matrix('path/to/Predictions', 'XGBoost')
