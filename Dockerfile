FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the contents of your project to the container
COPY . /app

# Update package lists and install Git, AWS CLI, and other dependencies
RUN apt-get update -y && \
    apt-get install -y git awscli ffmpeg libsm6 libxext6 unzip && \
    pip install -r requirements.txt

# Command to run your Flask application
CMD ["python3", "app.py"]
