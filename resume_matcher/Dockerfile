# Use a smaller base image with Python and pip
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (PDF, fonts, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download spaCy model ahead of time
RUN python -m spacy download en_core_web_sm

# Copy app code
COPY app ./app

# Expose port (Render uses PORT env var)
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
