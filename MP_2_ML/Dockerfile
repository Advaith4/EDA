# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install dependencies without saving pip cache (keeps image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API script and the trained model pipeline
COPY api.py .
COPY model.pkl .

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
