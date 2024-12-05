import tensorflow as tf
from groq import Groq
from fastapi import FastAPI
import os
from sklearn.preprocessing import LabelEncoder
from app.helper import load_json_item, get_majority_output, predict_disease

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Define constant
THRESHOLD = 5
MODEL_PATH = os.path.join(APP_DIR, '..', 'model', 'model.keras')

# Load symptoms and diseases into constant
SYMPTOMS = load_json_item("symptoms")
DISEASES = load_json_item("diseases")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)  # For H5 format use `load_model('my_model.h5')`

# Create FastAPI app
app = FastAPI()

# Create groq client
client = Groq(
    api_key=os.environ.get('GROQ_API_KEY')
)

# Create a LabelEncoder instance
label_encoder = LabelEncoder()
encoded_diseases = label_encoder.fit_transform(DISEASES)

@app.get('/')
def read_root():
    return {'message': 'Internal Disease Prediction'}

@app.post('/predict')
def predict(data: dict):
    """
    POST /predict

    Description:
        This endpoint predicts a disease based on user-provided symptoms and a user story.

    Request:
        Content-Type: application/json
        Body:
        {
            "user_story": "string"
        }
        - user_story (string, required): A textual description of symptoms or a user's health story.

    Response:
        Content-Type: application/json
        Body:
        {
            "predicted_disease": "string",
            "identified_symptoms": ["string", "string", ...]
        }
        - predicted_disease (string): The name of the predicted disease based on the identified symptoms.
        - identified_symptoms (array of strings): A list of symptoms identified from the `user_story`.

    Example:
        Request:
        POST /predict HTTP/1.1
        Host: example.com
        Content-Type: application/json

        {
            "user_story": "I have been experiencing severe headaches and occasional nausea."
        }

        Response:
        {
            "predicted_disease": "Migraine",
            "identified_symptoms": ["headache", "nausea"]
        }

    Errors:
        - 400 Bad Request: If the `user_story` field is missing or invalid.
        - 500 Internal Server Error: If an error occurs while processing the prediction.
    """
    user_symptoms = get_majority_output(data["user_story"], THRESHOLD, SYMPTOMS, client)
    disease_prediction = predict_disease(user_symptoms, model, label_encoder, SYMPTOMS)
    return {
        'predicted_disease': disease_prediction,
        'identified_symptoms': user_symptoms
        }

@app.post('/predict_manual')
def predict_manual(data: dict):
    """
    POST /predict_manual

    Description:
        This endpoint predicts a disease based on a list of user-provided symptoms.

    Request:
        Content-Type: application/json
        Body:
        {
            "symptoms": ["string", "string", ...]
        }
        - symptoms (array of strings, required): A list of symptoms provided by the user.

    Response:
        Content-Type: application/json
        Body:
        {
            "predicted_disease": "string"
        }
        - predicted_disease (string): The name of the predicted disease based on the provided symptoms.

    Example:
        Request:
        POST /predict_manual HTTP/1.1
        Host: example.com
        Content-Type: application/json

        {
            "symptoms": ["fever", "cough"]
        }

        Response:
        {
            "predicted_disease": "Flu"
        }

    Errors:
        - 400 Bad Request: If the `symptoms` field is missing or invalid.
        - 500 Internal Server Error: If an error occurs while processing the prediction.
    """
    disease_prediction = predict_disease(data["symptoms"], model, label_encoder, SYMPTOMS)
    return {
        'predicted_disease': disease_prediction
    }