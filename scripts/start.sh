#!/bin/bash

# Initialize the vector database
python -m scripts.init_vector_db

# Start the FastAPI application
exec uvicorn app:app --host 0.0.0.0 --port 80