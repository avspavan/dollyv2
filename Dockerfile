# Use the base image as the first stage
FROM python:3.8-slim as builder

# Install Git
RUN apt-get update && apt-get install -y git

# Install pip
RUN pip install --no-cache-dir --upgrade pip

# Set up a virtual environment
RUN python -m venv /opt/venv

# Make sure we use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt
# Copy the .whl package into the builder stage
COPY openvino_dev-2023.1.0.dev20230811-12050-py3-none-any.whl /app/

# Install the local .whl package using pip
RUN pip install /app/openvino_dev-2023.1.0.dev20230811-12050-py3-none-any.whl

# Use the builder stage as the base image for the final stage
FROM python:3.8-slim

# Copy the virtual environment from the builder stage to the final image
COPY --from=builder /opt/venv /opt/venv

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app/

# Set the entry point to activate the virtual environment and run server.py
CMD ["/opt/venv/bin/python", "server.py"]
