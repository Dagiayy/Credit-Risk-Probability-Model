# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements first and install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run FastAPI app when container starts
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
