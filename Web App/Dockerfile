# Base image
FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

# Install dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt

# Copy the application code and files
COPY . /app
WORKDIR /app

# Run the application on port 8080
EXPOSE 8080
CMD ["python", "app.py"]
