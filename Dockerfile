# Use a Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Streamlit and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application directory into the container
COPY . .

# Set the environment variable for the port
ENV PORT=8080

# Use CMD to start the FastAPI app
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
