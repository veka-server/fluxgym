#!/bin/bash

# Ensure Hugging Face models are downloaded if HF_HUB_OFFLINE is set to 0
if [ "$HF_HUB_OFFLINE" = "1" ]; then
    echo "Downloading Hugging Face models..."
    huggingface-cli download openai/clip-vit-large-patch14
    huggingface-cli download google/t5-v1_1-xxl
    huggingface-cli download MiaoshouAI/Florence-2-base-PromptGen-v2.0
else
    echo "HF_HUB_OFFLINE is set to 0, skipping model downloads."
fi

# Execute the main application
exec "$@"
