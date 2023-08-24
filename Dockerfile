# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the app code, models, and requirements into the container at /app
COPY . /app/

# Set the default model name as an environment variable
# models available: dolly-v2-3b ; llama7b
ENV DEFAULT_MODEL=llama7b

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose any ports your app might use
EXPOSE 8000

# Run your app when the container launches
CMD ["python", "server.py"]

