# Classifier Evaluation Functions

###################################################################################################

# Packages
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import os

###################################################################################################

def evaluate_classifier(y_true, y_pred, classifier_name):
    '''
    Input predicted labels and a classifier name, output a table of the accuracy, precision, recall, f1 score, and a plot of the confusion matrix.
    '''

    # Make the Output Directory if it doesn't exist
    if not os.path.exists('../../Output/Classifier Evaluation'):
        os.makedirs('../../Output/Classifier Evaluation')

    # Convert labels to integers
    label_mapping = {
        'SUV': 0,
        'Sedan': 1,
        'Pickup': 2,
        'Convertible': 3
    }
    y_true_num = np.array([label_mapping[label] for label in y_true])
    y_pred_num = np.array([label_mapping[label] for label in y_pred])

    # Print the classification report
    print(f"Classification Report for {classifier_name}:")
    print(classification_report(y_true_num, y_pred_num, target_names=label_mapping.keys()))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Confusion Matrix Display
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_mapping.keys())

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm_display.plot(cmap='Blues', ax=plt.gca(), xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('../../Output/Classifier Evaluation/' + classifier_name + '_Confusion_Matrix.png')
    plt.show()

