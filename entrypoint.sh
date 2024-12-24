#!/bin/bash

# Ensure Hugging Face models are downloaded if HF_HUB_OFFLINE is set to 0
if [ "$HF_HUB_OFFLINE" = "1" ]; then
    echo "Downloading Hugging Face models..."
#    HF_HUB_OFFLINE=0 huggingface-cli download openai/clip-vit-large-patch14
#    HF_HUB_OFFLINE=0 huggingface-cli download google/t5-v1_1-xxl t5xxl_fp16.safetensors
    HF_HUB_OFFLINE=0 huggingface-cli download MiaoshouAI/Florence-2-base-PromptGen-v2.0
    
#    HF_HUB_OFFLINE=0 huggingface-cli download cocktailpeanut/xulf-dev flux1-dev.sft --local-dir models/unet
#    HF_HUB_OFFLINE=0 huggingface-cli download cocktailpeanut/xulf-dev ae.sft --local-dir models/vae
#    HF_HUB_OFFLINE=0 huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/clip
#    HF_HUB_OFFLINE=0 huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/clip
#    HF_HUB_OFFLINE=0 huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/clip
    
    HF_HUB_OFFLINE=0 huggingface-cli download cocktailpeanut/xulf-dev flux1-dev.sft 
    HF_HUB_OFFLINE=0 huggingface-cli download cocktailpeanut/xulf-dev ae.sft 
    HF_HUB_OFFLINE=0 huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors 
    HF_HUB_OFFLINE=0 huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors 
    HF_HUB_OFFLINE=0 huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors 

    HF_HUB_OFFLINE=1
else
    echo "HF_HUB_OFFLINE is set to 0, skipping model downloads."
fi

# Execute the main application
exec "$@"
