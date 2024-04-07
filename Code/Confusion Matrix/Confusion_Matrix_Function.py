
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def visualize_classifier_confusion_matrix(base_path, classifier_name):
    classifier_dir = os.path.join(base_path, classifier_name)

    file_pattern = os.path.join(classifier_dir, f'{classifier_name.replace(" ", "_")}*.xlsx')
    files = glob.glob(file_pattern)

    if not files:
        print(f"No Excel files found for classifier '{classifier_name}' in '{classifier_dir}'.")
        return
    
    file_path = files[0]

    data = pd.read_excel(file_path)

    actual_labels = data['Class']
    predicted_column = [col for col in data.columns if 'Classification' in col][0]  # Dynamically find the predicted column
    predicted_labels = data[predicted_column]

    # Compute the confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels, labels=actual_labels.unique())

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=actual_labels.unique(), yticklabels=actual_labels.unique())
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title(f'Confusion Matrix for {classifier_name}')
    plt.show()

#Example usage of the function:
# visualize_classifier_confusion_matrix('path/to/Predictions', 'Logistic Regression')
# visualize_classifier_confusion_matrix('path/to/Predictions', 'XGBoost')
