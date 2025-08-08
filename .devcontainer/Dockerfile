# Use the official FEniCS Docker image as base
FROM dolfinx/dolfinx:stable

# Set environment variables to avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in one layer and clean up
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip in separate layer
RUN pip3 install --upgrade pip

# Set working directory early
WORKDIR /app

# Copy requirements.txt first for better layer caching
COPY requirements.txt .

# Install lightweight dependencies first (these change less frequently)
RUN pip3 install --no-cache-dir \
    contourpy==1.3.2 \
    cycler==0.12.1 \
    filelock==3.13.1 \
    fonttools==4.59.0 \
    fsspec==2024.6.1 \
    Jinja2==3.1.4 \
    kiwisolver==1.4.8 \
    MarkupSafe==2.1.5 \
    mpmath==1.3.0 \
    networkx==3.3 \
    packaging==25.0 \
    pyparsing==3.2.3 \
    python-dateutil==2.9.0.post0 \
    setuptools==70.2.0 \
    six==1.17.0 \
    sympy==1.13.3 \
    typing_extensions==4.12.2

# Install numpy and scipy (medium weight, math libraries)
RUN pip3 install --no-cache-dir \
    numpy>=2.2.6 \
    scipy>=1.15.0

# Install matplotlib and pillow (image processing)
RUN pip3 install --no-cache-dir \
    matplotlib==3.10.3 \
    pillow==11.3.0

# Install PyTorch last (heaviest - most likely to fail, so rebuild from here if needed)
RUN pip3 install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu129

# Default command
CMD ["bash"]