# Base image with CUDA 12.2
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Define environment variables for UID and GID and local timezone
ENV PUID=${PUID:-1000}
ENV PGID=${PGID:-1000}
ENV HF_HUB_OFFLINE=1 
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Install pip if not already installed
RUN apt-get update -y && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential  # Install dependencies for building extensions

# Create a group and user with the specified GID and UID
RUN groupadd -g "${PGID}" appuser ; useradd -m -s /bin/sh -u "${PUID}" -g "${PGID}" appuser

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chown appuser:appuser /entrypoint.sh; chmod +x /entrypoint.sh

WORKDIR /app

# Copy fluxgym application code
COPY . ./fluxgym

WORKDIR /app/fluxgym

# Get sd-scripts from kohya-ss
RUN git clone -b sd3 https://github.com/kohya-ss/sd-scripts ;

# install pip library
RUN pip install --no-cache-dir -r ./requirements.txt ; \
    cd sd-scripts ; \
    pip install --no-cache-dir -r ./requirements.txt ; \
    cd ../

# Install Torch, Torchvision, and Torchaudio for CUDA 12.2
RUN pip install huggingface_hub torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu122/torch_stable.html

RUN chown -R appuser:appuser /app

EXPOSE 7860

# use volume for cached model huggingface
VOLUME /home/appuser/.cache/huggingface/

USER appuser
WORKDIR /app/fluxgym

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Run fluxgym Python application
CMD ["python3", "./app.py"]
