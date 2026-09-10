# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only what the API needs
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY api/ ./api/
COPY models/ ./models/
COPY data/features.json ./data/features.json

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
