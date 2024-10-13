# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install system dependencies (including distutils)
RUN apt-get update && apt-get install -y \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
