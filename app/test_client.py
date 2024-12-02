# test_main.py
from fastapi.testclient import TestClient
from app.server import app  # Import the FastAPI app from your main.py file
from unittest.mock import patch

client = TestClient(app)  # Create a test client for the FastAPI app

def test_read_root():
    # Send a GET request to the root endpoint
    response = client.get("/")
    
    # Assert that the status code is 200
    assert response.status_code == 200
    
    # Assert the response JSON contains the expected message
    assert response.json() == {"message": "Internal Disease Prediction"}


def test_predict():
    # Mock the external functions and variables used in the predict endpoint
    with patch("server.get_majority_output") as mock_get_majority_output, \
         patch("server.predict_disease") as mock_predict_disease, \
         patch("server.model") as mock_model, \
         patch("server.label_encoder") as mock_label_encoder:

        # Mock the return values of the functions
        mock_get_majority_output.return_value = ['fever', 'headache']  # mocked user symptoms
        mock_predict_disease.return_value = "Flu"  # mocked disease prediction

        # Send a POST request with a user story
        response = client.post("/predict", json={"user_story": "I have fever and headache"})
        
        # Assert that the status code is 200
        assert response.status_code == 200
        