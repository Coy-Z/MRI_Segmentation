# Use the official FEniCS Docker image as base
FROM dolfinx/dolfinx:stable

# Set environment variables to avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install pip and Python build dependencies
RUN apt-get update && apt-get install -y python3-pip python3-dev

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy your requirements.txt into the container
COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Install dependencies from requirements.txt
# We'll handle torch and related packages separately to specify CUDA version
RUN pip3 install --no-cache-dir -r requirements.txt

# Install specific CUDA PyTorch packages
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Copy the rest of your repo
COPY . /app

# Default command (modify as needed)
CMD ["bash"]
