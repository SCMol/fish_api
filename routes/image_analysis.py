import tempfile
import requests
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO

# Download the model and save it to a temporary file
model_url = "https://storage.googleapis.com/fresh_bucket_1557/predict_models/salmon_best_8mar24.h5"
response = requests.get(model_url)

with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
    tmp.write(response.content)
    tmp.flush()
    model = load_model(tmp.name)

def preprocess_image(image: Image.Image):
    image = image.resize((150, 150))
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    return image_array

def interpret_prediction(prediction):
    # Assuming 'prediction' is a NumPy array
    class_label = 'healthy' if prediction[0][0] < 0.5 else 'unhealthy'
    probability = float(prediction[0][0])  # Convert to native Python float
    return class_label, probability

def analyze_image(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        processed_image = preprocess_image(image)
        prediction = model.predict(np.array([processed_image]))
        class_label, probability = interpret_prediction(prediction)
        return {"class_label": class_label, "probability": probability}
    except Exception as e:
        print("Error during interpretation of prediction:", str(e))
        print("Prediction:", prediction)
        raise
