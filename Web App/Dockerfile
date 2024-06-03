# Base image
FROM pytorch/pytorch:latest

# Install dependencies
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

# Copy the application code and files
COPY . /app
WORKDIR /app

# Run the application on port 5000
EXPOSE 5000
CMD ["python", "app.py"]
