import tempfile
import requests
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO

# Download the model and save it to a temporary file
model_url = 'https://storage.googleapis.com/fresh_bucket_1557/predict_models/alex_model3_8mar24.h5'
response = requests.get(model_url)

with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
    tmp.write(response.content)
    tmp.flush()
    species_id_model = load_model(tmp.name)

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    return image_array

def interpret_species_id(prediction):
    # Adapt this function to match the output of your species identification model
    species_names = ['Species A', 'Species B', 'Species C',
    'Species D', 'Species E', 'Species F',
    'Species G', 'Species H', 'Species I',
    'Species J']
    predicted_species_index = np.argmax(prediction)
    species_label = species_names[predicted_species_index]
    return species_label

def id_species(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        processed_image = preprocess_image(image)
        prediction = species_id_model.predict(np.array([processed_image]))
        species_label = interpret_species_id(prediction)
        return species_label
    except Exception as e:
        print("Error during species identification:", str(e))
        raise
