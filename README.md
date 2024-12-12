# Google Bangkit 2024 Capstone Machine Learning Model
Insight - Machine Learning Model and Prediction API

## InSight Prediction API Documentation
| Method   | Endpoint     | Summary    | Description | Request Body | Response |
| ------------ | ------------ | ------------ | --------| ---------| -----------
| `POST` | /predict | Predict disease based on user story | Predicts the disease based on the provided user story. | Query Param: `user_story` | `predicted_disease` : string |
| `POST` | /predict_manual| Predict disease based on symptoms | Predicts the disease based on the provided symptoms. | Query Param: `symptoms` | `predicted_disease` : string |

## Project Setup
Follow these steps to setup and run the application:

1. **Copy the .env structure to create your development environment file:**
```
cp .env.example .env.dev
```
Update the values in `.env.dev` as needed.

2. **Install the dependencies:**
```
pip install -r requirement.txt
```
Run the command from the root directory

3. **Run the server:**
```
uvicorn app.server:app --reload
```
Run the command from the root directory, server will run on `localhost:8000` and you can open `localhost:8000/doc` to access the swagger UI for API testing.

## Credits
This project is implemented by Team C242-PS463.