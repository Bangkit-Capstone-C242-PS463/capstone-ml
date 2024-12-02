# import tensorflow as tf
# import numpy as np
# from groq import Groq
# from fastapi import FastAPI
# import os
# import json
# import math
# from collections import Counter
# from sklearn.preprocessing import LabelEncoder

# # Define constant
# THRESHOLD = 5
# MODEL_PATH = 'model.keras'

# # Load the trained model
# model = tf.keras.models.load_model(MODEL_PATH)  # For H5 format use `load_model('my_model.h5')`

# # Create FastAPI app
# app = FastAPI()

# # Create groq client
# client = Groq(
#     api_key=os.environ.get('GROQ_API_KEY')
# )

# def load_json_item(column: str, file_path='diseases_and_symptoms.json') -> list[str]:
#     """
#     load data from json file to a list

#     Parameters:
#     column (str): name of column to be extract
#     file_path (str): path of the file 

#     Returns:
#     list of str: list of items that extracted
#     """
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     return data[column]

# # Load symptoms and diseases into constant
# SYMPTOMS = load_json_item("symptoms")
# DISEASES = load_json_item("diseases")

# # Create a LabelEncoder instance
# label_encoder = LabelEncoder()
# encoded_diseases = label_encoder.fit_transform(DISEASES)

# def process_string_input(input: str) -> list[str]:
#     """
#     process an input (user's story) into a list of symptoms using groq api llm's

#     Parameters:
#     input (str): input string to be identified (user's story)
    
#     Returns:
#     list of str: list of identified symptoms
#     """
#     chat_completion = client.chat.completions.create(
#         #
#         # Required parameters
#         #
#         messages=[
#             # Set an optional system message. This sets the behavior of the
#             # assistant and can be used to provide specific instructions for
#             # how it should behave throughout the conversation.
#             {
#                 "role": "system",
#                 "content": f"""You are an assistant who helps the hospital to map symptoms from a patient's story, a collection of known symptoms are as follows, {SYMPTOMS}

#                 please consider all the symptoms in the list above when creating an ouput, output only the symptoms in the prompt without any additional message.
#                 """
#             },
#             # Set a user message for the assistant to respond to.
#             {
#                 "role": "user",
#                 "content": input,
#             }
#         ],

#         # The language model which will generate the completion.
#         model="llama-3.2-90b-vision-preview",

#         #
#         # Optional parameters
#         #

#         # Controls randomness: lowering results in less random completions.
#         # As the temperature approaches zero, the model will become deterministic
#         # and repetitive.
#         temperature=0.5,

#         # The maximum number of tokens to generate. Requests can use up to
#         # 32,768 tokens shared between prompt and completion.
#         max_tokens=1024,

#         # Controls diversity via nucleus sampling: 0.5 means half of all
#         # likelihood-weighted options are considered.
#         top_p=1,

#         # A stop sequence is a predefined or user-specified text string that
#         # signals an AI to stop generating content, ensuring its responses
#         # remain focused and concise. Examples include punctuation marks and
#         # markers like "[end]".
#         stop=None,

#         # If set, partial message deltas will be sent.
#         stream=False,
#     )

#     # Return the completion message
#     llm_output = chat_completion.choices[0].message.content
#     symptoms_list = [symptom.strip() for symptom in llm_output.split(',')]
#     return symptoms_list

# def get_majority_output(input: str, threshold: int) -> list[str]:
#     """
#     process the input with THRESHOLD amount of times to get the majority output.
#     this is used to handle the llm's hallucination problem.

#     Parameters:
#     input (str): input string to be identified (user's story)
#     threshold (int): how many times to make request to groq API

#     Returns:
#     list of str: list of final symptoms that appear at least ceil(threshold/2) times
#     """
#     minimum = math.ceil(threshold/2)
#     all_llm_output_symptoms = []
#     final_symptoms = []
#     for i in range(threshold):
#         llm_output_symptoms = process_string_input(input)
#         all_llm_output_symptoms.append(llm_output_symptoms)
    
#     flattened_symptoms = [symptom.lower() for sublist in all_llm_output_symptoms for symptom in sublist]
#     symptom_count = Counter(flattened_symptoms)

#     for symptom, count in symptom_count.items():
#         # check if majority of the prediction outputed the symptom (met the minimum) and the symptom is a valid symptom in the list
#         if count >= minimum and symptom in SYMPTOMS:
#             final_symptoms.append(symptom)

#     return final_symptoms

# def one_hot_encode_symptoms(symptoms_list, all_symptoms) -> list[int]:
#        """
#        process a list of symptoms into one hot encoded representation relative to all the available symptoms

#        Parameters:
#        symptoms_list (list of str): list of symptoms that want to be transform
#        all_symptoms (list of str): all the available symptoms in the dataset

#        Returns: 
#        list of int: one hot encoded representation of the symptoms
#        """
#        encoded_symptoms = [0] * len(all_symptoms)  # Initialize with zeros
#        for symptom in symptoms_list:
#            if symptom in all_symptoms:
#                index = all_symptoms.index(symptom)
#                encoded_symptoms[index] = 1
#        return encoded_symptoms

# def predict_disease(input_symptoms: list[str], model: tf.keras.Model, label_encoder: LabelEncoder, all_symptoms: list[str]) -> str:
#     """
#     predict the disease based on the input symptoms

#     Parameters:
#     input_symptoms (str): list of symptoms to predict the disease
#     model (tf.keras.Model): a trained tensorflow model used for disease prediction
#     label_encoder (LabelEncoder): A fitted LabelEncoder instance used to decode the predicted label back into the disease name
#     all_symptoms (list of str): list of all possible symptoms for one hot encoding

#     Returns:
#     str: the name of the predicted disease
#     """
#     encoded_input = one_hot_encode_symptoms(input_symptoms, all_symptoms)
#     input_tensor = tf.convert_to_tensor([encoded_input], dtype=tf.float32)
#     input_dataset = tf.data.Dataset.from_tensor_slices(input_tensor).batch(1)
#     predictions = model.predict(input_dataset)
#     predicted_label = predictions.argmax()
#     predicted_disease = label_encoder.inverse_transform([predicted_label])[0]
#     return predicted_disease


# @app.get('/')
# def read_root():
#     return {'message': 'Internal Disease Prediction'}

# @app.post('/predict')
# def predict(data: dict):
#     user_symptoms = get_majority_output(data["user_story"], THRESHOLD)
#     disease_prediction = predict_disease(user_symptoms, model, label_encoder, SYMPTOMS)
#     return {
#         'predicted_disease': disease_prediction,
#         'identified_symptoms': user_symptoms
#         }

# @app.post('/predict_manual')
# def predict_manual(data: dict):
#     disease_prediction = predict_disease(data["symptoms"], model, label_encoder, SYMPTOMS)
#     return {
#         'predicted_disease': disease_prediction
#     }
    