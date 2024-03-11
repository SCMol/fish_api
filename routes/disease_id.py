import requests
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO
import tempfile

class_names = ['Bacterial diseases', 'Fungal diseases', 'Parasitic diseases']

model_url = "https://storage.googleapis.com/fresh_bucket_1557/predict_models/disease_id.h5"
response = requests.get(model_url)

with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
    tmp.write(response.content)
    tmp.flush()
    model = load_model(tmp.name)

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    return image_array

def interpret_class_id(prediction, class_names):
    # Existing logic to determine the predicted class index
    predicted_class_index = np.argmax(prediction)
    # Ensure only two values are returned: class_label and a probability value
    class_label = class_names[predicted_class_index]
    probability = prediction[0][predicted_class_index]  # assuming prediction is a 2D array

    return class_label, probability

def analyze_disease(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        processed_image = preprocess_image(image)
        prediction = model.predict(np.array([processed_image]))
        class_label, probability = interpret_class_id(prediction, class_names)

        # Convert numpy.float32 to native Python float
        probability = float(probability)

        return {"class_label": class_label, "probability": probability}
    except Exception as e:
        print(f"Error during interpretation of prediction: {str(e)}")
        print("Prediction:", prediction)
        raise
