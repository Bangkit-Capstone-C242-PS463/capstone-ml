# capstone-ml

## How to Run
1. Install all the dependencies using by running the command `pip install -r requirements.txt` from the root directory
2. Make the copy of the .env file, put your own Groq API key
3. To run the server, run this command `uvicorn app.server:app --reload`
4. Server will run on `localhost:8000`, you can open `localhost:8000/docs` for testing