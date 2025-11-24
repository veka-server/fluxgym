# Base image with CUDA 12.2
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Install pip if not already installed
RUN apt-get update -y && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0
#    build-essential  # Install dependencies for building extensions

# Define environment variables for UID and GID and local timezone
ENV PUID=${PUID:-1000}
ENV PGID=${PGID:-1000}

# Create a group with the specified GID
#RUN groupadd -g "${PGID}" appuser
# Create a user with the specified UID and GID
#RUN useradd -m -s /bin/sh -u "${PUID}" -g "${PGID}" appuser

WORKDIR /app

# Get sd-scripts from kohya-ss and install them
RUN git clone -b sd3 https://github.com/kohya-ss/sd-scripts && \
    cd sd-scripts && \
    pip install --no-cache-dir -r ./requirements.txt

# Install main application dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt

# Install Torch, Torchvision, and Torchaudio for CUDA 12.2
# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu122/torch_stable.html
#RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu122/torch_stable.html
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN chown -R appuser:appuser /app

# delete redundant requirements.txt and sd-scripts directory within the container
#RUN rm -r ./sd-scripts
#RUN rm ./requirements.txt

#Run application as non-root
#USER appuser

# Copy fluxgym application code
COPY . ./fluxgym

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

# Installer huggingface-cli
RUN pip install --no-cache-dir huggingface_hub

#USER appuser
WORKDIR /home/appuser

# Téléchargement des modèles en mode HF_HUB_OFFLINE=0
RUN huggingface-cli download openai/clip-vit-large-patch14 && \
    huggingface-cli download google/t5-v1_1-xxl

# Ajouter la variable d'environnement pour le mode offline
ENV HF_HUB_OFFLINE=1

# Run fluxgym Python application
CMD ["python3", "./app.py"]
