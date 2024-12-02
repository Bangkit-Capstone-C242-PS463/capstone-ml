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
    user_symptoms = get_majority_output(data["user_story"], THRESHOLD, SYMPTOMS, client)
    disease_prediction = predict_disease(user_symptoms, model, label_encoder, SYMPTOMS)
    return {
        'predicted_disease': disease_prediction,
        'identified_symptoms': user_symptoms
        }

@app.post('/predict_manual')
def predict_manual(data: dict):
    disease_prediction = predict_disease(data["symptoms"], model, label_encoder, SYMPTOMS)
    return {
        'predicted_disease': disease_prediction
    }