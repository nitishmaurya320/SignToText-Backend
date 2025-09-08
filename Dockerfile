# Base image: Python 3.12
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (opencv, mediapipe, etc. need these)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose port (Render will override this automatically)
EXPOSE 5000

# Start the Flask app
CMD ["python", "server.py"]
