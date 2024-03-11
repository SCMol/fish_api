import tempfile
import requests
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO
import re  # Importing re for regular expressions
from sumy.parsers.plaintext import PlaintextParser  # Importing PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # Importing LsaSummarizer

# Download the model and save it to a temporary file
model_url = 'https://storage.googleapis.com/fresh_bucket_1557/predict_models/species_id.h5'
response = requests.get(model_url)

class_to_wikipedia_page = {
    'Gilt Head Bream': 'Sparus_aurata',
    'Red Sea Bream': 'Pagrus_major',
    'Sea Bass': 'Bass_(fish)',
    'Red Mullet': 'Mullus_barbatus',
    'Horse Mackerel': 'Trachurus',
    'Black Sea Sprat': 'Sprat',
    'Striped Red Mullet': 'Mullus_surmuletus',
    'Trout': 'Trout',
    'Shrimp': 'Shrimp'
}

class_names = [
    'Gilt Head Bream',
    'Red Sea Bream',
    'Sea Bass',
    'Red Mullet',
    'Horse Mackerel',
    'Black Sea Sprat',
    'Striped Red Mullet',
    'Trout',
    'Shrimp'
]

with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
    tmp.write(response.content)
    tmp.flush()
    class_id_model = load_model(tmp.name)

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    return image_array

def interpret_class_id(prediction, class_names):
    # Adapt this function to match the output of your class identification model
    predicted_class_index = np.argmax(prediction)
    class_label = class_names[predicted_class_index]
    return class_label

def search_wikipedia(query):
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": query
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        page_id = next(iter(data["query"]["pages"].keys()), "-1")
        if page_id != "-1":
            extract = data["query"]["pages"][page_id]["extract"]
            return extract
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error making Wikipedia API request: {str(e)}")
        return None
    except Exception as e:
        print(f"Error retrieving Wikipedia information: {str(e)}")
        return None

def clean_answer(answer):
    clean_text = re.sub(r"\s+", " ", answer)
    return clean_text.strip()

def summarize_answer(answer, num_sentences=5):
    parser = PlaintextParser.from_string(answer, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

def id_class(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        processed_image = preprocess_image(image)
        prediction = class_id_model.predict(np.array([processed_image]))
        class_label = interpret_class_id(prediction, class_names)
        return class_label
    except Exception as e:
        print(f"Error during interpretation of prediction: {str(e)}")
        print("Prediction:", prediction)
        raise
