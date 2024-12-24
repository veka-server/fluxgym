# Base image with CUDA 12.2
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Define environment variables for UID and GID and local timezone
ENV PUID=${PUID:-1000}
ENV PGID=${PGID:-1000}
ENV HF_HUB_OFFLINE=0
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Install pip if not already installed
RUN apt-get update -y && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential  # Install dependencies for building extensions

# Create a group and user with the specified GID and UID
RUN groupadd -g "${PGID}" appuser ; useradd -m -s /bin/sh -u "${PUID}" -g "${PGID}" appuser

WORKDIR /app

# Get sd-scripts from kohya-ss and install them
RUN git clone -b sd3 https://github.com/kohya-ss/sd-scripts && \
    cd sd-scripts && \
    pip install --no-cache-dir -r ./requirements.txt

# Install main application dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

# Install Torch, Torchvision, and Torchaudio for CUDA 12.2
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu122/torch_stable.html

RUN pip install flash_attn

RUN chown -R appuser:appuser /app

# delete redundant requirements.txt and sd-scripts directory within the container
RUN rm -r ./sd-scripts
RUN rm ./requirements.txt

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chown appuser:appuser /entrypoint.sh; chmod +x /entrypoint.sh

#Run application as non-root
USER appuser

# Copy fluxgym application code
COPY . ./fluxgym

WORKDIR /app/fluxgym

USER root
RUN git clone -b sd3 https://github.com/kohya-ss/sd-scripts
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

# use volume for cached model huggingface
VOLUME /home/appuser/.cache/huggingface/

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Run fluxgym Python application
CMD ["python3", "./app.py"]
