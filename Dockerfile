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
COPY models_v2/ ./models_v2/
COPY data/features.json ./data/features.json
COPY data/features_apc_adc_v2.json ./data/features_apc_adc_v2.json
COPY data/features_ndc_v2.json ./data/features_ndc_v2.json

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
