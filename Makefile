ci:
	pip install -r requirements.txt
act-venv:
	source venv/bin/activate
start-service:
	uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload