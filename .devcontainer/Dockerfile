FROM dolfinx/dolfinx:stable

# Environment (rarely changes) - Layer 1
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies (rarely changes) - Layer 2
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && pip3 install --upgrade pip

WORKDIR /app

# Copy requirements first (changes occasionally) - Layer 3
COPY requirements.txt .

# Stable utility dependencies (change rarely) - Layer 4
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
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

# Math libraries (change occasionally) - Layer 5
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    numpy>=2.2.6 \
    scipy>=1.15.0

# Image processing libraries - Layer 6
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    matplotlib==3.10.3 \
    pillow==11.3.0

# Heavy ML dependencies (most likely to fail/change) - Layer 7
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu129

CMD ["bash"]