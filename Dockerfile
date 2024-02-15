# Use the PyTorch image as the base
FROM nvcr.io/nvidia/jax:23.10-py3

# Set the working directory in the container to /workspace
WORKDIR /workspace

# Install Python, Jupyter, and Git
RUN apt-get update && \
	apt-get install -y python3-pyqt5 libqt5widgets5 x11-apps && \
	apt-get install -y git && \
    pip install --no-cache-dir jupyter && \
    rm -rf /var/lib/apt/lists/*

# (Optional) If you have additional dependencies in a requirements.txt file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

