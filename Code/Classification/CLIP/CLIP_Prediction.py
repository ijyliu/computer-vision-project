
from autogluon.multimodal import MultiModalPredictor
from IPython.display import Image

# Function to take an input filepath and return the CLIP probabilities
def get_clip_prediction(filepath):

    # Load image from filepath
    loaded_img = Image(filename=filepath)

    # Get CLIP prediction
    predictor = MultiModalPredictor(problem_type="zero_shot_image_classification")
    probs = predictor.predict_proba({"image": [loaded_img]}, {"text": ['This is an SUV', 'This is a Sedan', 'This is a Pickup', 'This is a Convertible']})

    # Return probabilities
    return probs
