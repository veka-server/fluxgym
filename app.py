import os
import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'
sys.path.insert(0, os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))
sys.path.insert(1, 'sd-scripts')
import subprocess
import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM
from gradio_logsview import LogsView, LogsViewRunner
from huggingface_hub import hf_hub_download, HfApi
from library import flux_train_utils, huggingface_util
from argparse import Namespace
import train_network
import toml
import re
import base64
import urllib.request
import urllib.error

MAX_IMAGES = 150

with open('models.yaml', 'r') as file:
    models = yaml.safe_load(file)

def readme(base_model, lora_name, instance_prompt, sample_prompts):
    # ... (fonction inchangée) ...
    model_config = models[base_model]
    model_file = model_config["file"]
    base_model_name = model_config["base"]
    license = None
    license_name = None
    license_link = None
    license_items = []
    if "license" in model_config:
        license = model_config["license"]
        license_items.append(f"license: {license}")
    if "license_name" in model_config:
        license_name = model_config["license_name"]
        license_items.append(f"license_name: {license_name}")
    if "license_link" in model_config:
        license_link = model_config["license_link"]
        license_items.append(f"license_link: {license_link}")
    license_str = "\n".join(license_items)
    
    tags = [ "text-to-image", "flux", "lora", "diffusers", "template:sd-lora", "fluxgym" ]
    
    widgets = []
    sample_image_paths = []
    output_name = slugify(lora_name)
    samples_dir = resolve_path_without_quotes(f"outputs/{output_name}/sample")
    try:
        for filename in os.listdir(samples_dir):
            match = re.search(r"_(\d+)_(\d+)_(\d+)\.png$", filename)
            if match:
                steps, index, timestamp = int(match.group(1)), int(match.group(2)), int(match.group(3))
                sample_image_paths.append((steps, index, f"sample/{filename}"))
        sample_image_paths.sort(key=lambda x: x[0], reverse=True)
        final_sample_image_paths = sample_image_paths[:len(sample_prompts)]
        final_sample_image_paths.sort(key=lambda x: x[1])
        for i, prompt in enumerate(sample_prompts):
            _, _, image_path = final_sample_image_paths[i]
            widgets.append({"text": prompt, "output": {"url": image_path}})
    except:
        pass
    
    readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model_name}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
{license_str}
---

# {lora_name}

A Flux LoRA trained on a local computer with [Fluxgym](https://github.com/cocktailpeanut/fluxgym )

<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, Forge, etc.

Weights for this model are available in Safetensors format.
"""
    return readme_content

def account_hf():
    try:
        with open("HF_TOKEN", "r") as file:
            token = file.read()
            api = HfApi(token=token)
            try:
                account = api.whoami()
                return { "token": token, "account": account['name'] }
            except:
                return None
    except:
        return None

def logout_hf():
    os.remove("HF_TOKEN")
    global current_account
    current_account = account_hf()
    return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)

def login_hf(hf_token):
    api = HfApi(token=hf_token)
    try:
        account = api.whoami()
        if account != None and "name" in account:
            with open("HF_TOKEN", "w") as file:
                file.write(hf_token)
            global current_account
            current_account = account_hf()
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(value=current_account["account"], visible=True)
        return gr.update(), gr.update(), gr.update(), gr.update()
    except:
        return gr.update(), gr.update(), gr.update(), gr.update()

def upload_hf(base_model, lora_rows, repo_owner, repo_name, repo_visibility, hf_token):
    src = lora_rows
    repo_id = f"{repo_owner}/{repo_name}"
    gr.Info(f"Uploading to Huggingface. Please Stand by...", duration=None)
    args = Namespace(
        huggingface_repo_id=repo_id,
        huggingface_repo_type="model",
        huggingface_repo_visibility=repo_visibility,
        huggingface_path_in_repo="",
        huggingface_token=hf_token,
        async_upload=False
    )
    huggingface_util.upload(args=args, src=src)
    gr.Info(f"[Upload Complete] https://huggingface.co/{repo_id}", duration=None)

def load_captioning(uploaded_files, concept_sentence):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    
    if len(uploaded_images) <= 1:
        raise gr.Error("Please upload at least 2 images to train your model (the ideal number with default settings is between 4-30)")
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"For now, only {MAX_IMAGES} or less images are allowed for training")
    
    updates = []
    # 1. Captioning area visibility
    updates.append(gr.update(visible=True))
    
    # 2. Image rows, images, and captions
    for i in range(1, MAX_IMAGES + 1):
        visible = i <= len(uploaded_images)
        image_value = uploaded_images[i - 1] if visible else None
        
        # Row visibility
        updates.append(gr.update(visible=visible))
        # Image
        updates.append(gr.update(value=image_value, visible=visible))
        # Caption
        text_value = None
        if image_value:
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r') as file:
                    text_value = file.read()
            elif concept_sentence:
                text_value = concept_sentence
        updates.append(gr.update(value=text_value, visible=visible))
    
    # 3. Refresh button visibility
    updates.append(gr.update(visible=True))
    # 4. Start button visibility
    updates.append(gr.update(visible=True))
    
    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False)

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size/width) * height)
        else:
            new_height = size
            new_width = int((size/height) * width)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(destination_folder, size, *inputs):
    images = inputs[0]
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, image in enumerate(images):
        new_image_path = shutil.copy(image, destination_folder)
        ext = os.path.splitext(new_image_path)[-1].lower()
        if ext == '.txt':
            continue

        resize_image(new_image_path, new_image_path, size)
        original_caption = inputs[index + 1]
        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = resolve_path_without_quotes(os.path.join(destination_folder, caption_file_name))
        
        if os.path.exists(caption_path):
            print(f"{caption_path} already exists. using the existing .txt file")
        else:
            with open(caption_path, 'w') as file:
                file.write(original_caption)
    
    return destination_folder

def get_captioning_backend():
    """Vérifie si l'API OpenAI compatible est configurée"""
    openai_url = os.getenv("OPENAI_URL")
    openai_key = os.getenv("OPENAI_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    return all([openai_url, openai_key, openai_model])

def run_captioning(images, concept_sentence, *captions):
    print(f"run_captioning - concept: {concept_sentence}")
    
    use_openai = get_captioning_backend()
    
    if use_openai:
        print(f"✅ Using OpenAI compatible API (Model: {os.getenv('OPENAI_MODEL')})")
        openai_url = os.getenv("OPENAI_URL").rstrip('/')
        openai_key = os.getenv("OPENAI_KEY")
        openai_model = os.getenv("OPENAI_MODEL")
        
        if not openai_url.endswith('/chat/completions'):
            if not openai_url.endswith('/'):
                openai_url += '/'
            openai_url += 'v1/chat/completions'
    else:
        print("🤖 Using Florence-2 for captioning")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)
        processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    captions = list(captions)
    for i, image_path in enumerate(images):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")

        if use_openai:
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                image_format = Image.open(image_path).format.lower()
                mime_type = "image/png" if image_format == 'png' else "image/jpeg"
                
                payload = {
                    "model": openai_model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Generate a detailed caption for this image"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                            }
                        ]
                    }],
                    "max_tokens": 1024
                }
                
                headers = {
                    'Authorization': f'Bearer {openai_key}',
                    'Content-Type': 'application/json'
                }
                
                request = urllib.request.Request(
                    openai_url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers=headers
                )
                
                with urllib.request.urlopen(request) as response:
                    response_data = json.loads(response.read().decode('utf-8'))
                
                caption_text = response_data['choices'][0]['message']['content']
                
            except urllib.error.HTTPError as e:
                error_message = e.read().decode('utf-8')
                print(f"❌ OpenAI API error: {e.code} - {error_message}")
                caption_text = f"Error: HTTP {e.code}"
            except Exception as e:
                print(f"❌ OpenAI API error: {e}")
                caption_text = "Error generating caption"
        else:
            prompt = "<DETAILED_CAPTION>"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=1024, num_beams=3
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, task=prompt, image_size=(image.width, image.height)
            )
            caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        
        if concept_sentence:
            caption_text = f"{concept_sentence} {caption_text}"
        captions[i] = caption_text
        yield captions
    
    if not use_openai:
        model.to("cpu")
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def download(base_model):
    model = models[base_model]
    model_file = model["file"]
    repo = model["repo"]

    if base_model == "flux-dev" or base_model == "flux-schnell":
        unet_folder = "models/unet"
    else:
        unet_folder = f"models/unet/{repo}"
    unet_path = os.path.join(unet_folder, model_file)
    if not os.path.exists(unet_path):
        os.makedirs(unet_folder, exist_ok=True)
        gr.Info(f"Downloading base model: {base_model}. Please wait.", duration=None)
        hf_hub_download(repo_id=repo, local_dir=unet_folder, filename=model_file)

    vae_folder = "models/vae"
    vae_path = os.path.join(vae_folder, "ae.sft")
    if not os.path.exists(vae_path):
        os.makedirs(vae_folder, exist_ok=True)
        hf_hub_download(repo_id="cocktailpeanut/xulf-dev", local_dir=vae_folder, filename="ae.sft")

    clip_folder = "models/clip"
    clip_l_path = os.path.join(clip_folder, "clip_l.safetensors")
    if not os.path.exists(clip_l_path):
        os.makedirs(clip_folder, exist_ok=True)
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="clip_l.safetensors")

    t5xxl_path = os.path.join(clip_folder, "t5xxl_fp16.safetensors")
    if not os.path.exists(t5xxl_path):
        hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", local_dir=clip_folder, filename="t5xxl_fp16.safetensors")

def resolve_path(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return f"\"{norm_path}\""

def resolve_path_without_quotes(p):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.normpath(os.path.join(current_dir, p))
    return norm_path

def gen_sh(base_model, output_name, resolution, seed, workers, learning_rate, network_dim,
           max_train_epochs, save_every_n_epochs, timestep_sampling, guidance_scale, vram,
           sample_prompts, sample_every_n_steps, *advanced_components):
    
    print(f"gen_sh: network_dim:{network_dim}, epochs:{max_train_epochs}, save_every_n:{save_every_n_epochs}")
    
    output_dir = resolve_path(f"outputs/{output_name}")
    sample_prompts_path = resolve_path(f"outputs/{output_name}/sample_prompts.txt")
    
    line_break = "\\" if sys.platform != "win32" else "^"
    
    sample = ""
    if len(sample_prompts) > 0 and sample_every_n_steps > 0:
        sample = f"--sample_prompts={sample_prompts_path} --sample_every_n_steps=\"{sample_every_n_steps}\" {line_break}\n"
    
    if vram == "16G":
        optimizer = f"--optimizer_type adafactor {line_break}\n  --optimizer_args \"relative_step=False\" \"scale_parameter=False\" \"warmup_init=False\" {line_break}\n  --lr_scheduler constant_with_warmup {line_break}\n  --max_grad_norm 0.0 {line_break}\n"
    elif vram == "12G":
        optimizer = f"--optimizer_type adafactor {line_break}\n  --optimizer_args \"relative_step=False\" \"scale_parameter=False\" \"warmup_init=False\" {line_break}\n  --split_mode {line_break}\n  --network_args \"train_blocks=single\" {line_break}\n  --lr_scheduler constant_with_warmup {line_break}\n  --max_grad_norm 0.0 {line_break}\n"
    else:
        optimizer = f"--optimizer_type adamw8bit {line_break}\n"
    
    model_config = models[base_model]
    model_file = model_config["file"]
    repo = model_config["repo"]
    model_folder = "models/unet" if base_model in ["flux-dev", "flux-schnell"] else f"models/unet/{repo}"
    model_path = os.path.join(model_folder, model_file)
    pretrained_model_path = resolve_path(model_path)
    
    clip_path = resolve_path("models/clip/clip_l.safetensors")
    t5_path = resolve_path("models/clip/t5xxl_fp16.safetensors")
    ae_path = resolve_path("models/vae/ae.sft")
    
    sh = f"""accelerate launch {line_break}
  --mixed_precision bf16 {line_break}
  --num_cpu_threads_per_process 1 {line_break}
  sd-scripts/flux_train_network.py {line_break}
  --pretrained_model_name_or_path {pretrained_model_path} {line_break}
  --clip_l {clip_path} {line_break}
  --t5xxl {t5_path} {line_break}
  --ae {ae_path} {line_break}
  --cache_latents_to_disk {line_break}
  --save_model_as safetensors {line_break}
  --sdpa --persistent_data_loader_workers {line_break}
  --max_data_loader_n_workers {workers} {line_break}
  --seed {seed} {line_break}
  --gradient_checkpointing {line_break}
  --mixed_precision bf16 {line_break}
  --save_precision bf16 {line_break}
  --network_module networks.lora_flux {line_break}
  --network_dim {network_dim} {line_break}
  {optimizer}{sample}
  --learning_rate {learning_rate} {line_break}
  --cache_text_encoder_outputs {line_break}
  --cache_text_encoder_outputs_to_disk {line_break}
  --fp8_base {line_break}
  --highvram {line_break}
  --max_train_epochs {max_train_epochs} {line_break}
  --save_every_n_epochs {save_every_n_epochs} {line_break}
  --dataset_config {resolve_path(f"outputs/{output_name}/dataset.toml")} {line_break}
  --output_dir {resolve_path(f"outputs/{output_name}")} {line_break}
  --output_name {output_name} {line_break}
  --timestep_sampling {timestep_sampling} {line_break}
  --discrete_flow_shift 3.1582 {line_break}
  --model_prediction_type raw {line_break}
  --guidance_scale {guidance_scale} {line_break}
  --loss_type l2 {line_break}"""
    
    global advanced_component_ids, original_advanced_component_values
    advanced_flags = []
    for i, current_value in enumerate(advanced_components):
        if original_advanced_component_values[i] != current_value:
            if current_value == True:
                advanced_flags.append(advanced_component_ids[i])
            else:
                advanced_flags.append(f"{advanced_component_ids[i]} {current_value}")
    
    if advanced_flags:
        advanced_flags_str = f" {line_break}\n  ".join(advanced_flags)
        sh = sh + "\n  " + advanced_flags_str
    
    return sh

def gen_toml(dataset_folder, resolution, class_tokens, num_repeats):
    return f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{resolve_path_without_quotes(dataset_folder)}'
  class_tokens = '{class_tokens}'
  num_repeats = {num_repeats}"""

def update_total_steps(max_train_epochs, num_repeats, images):
    try:
        num_images = len(images)
        total_steps = max_train_epochs * num_images * num_repeats
        return gr.update(value=total_steps)
    except:
        return gr.update()

def set_repo(lora_rows):
    return gr.update(value=os.path.basename(lora_rows))

def get_loras():
    try:
        outputs_path = resolve_path_without_quotes(f"outputs")
        files = os.listdir(outputs_path)
        folders = [os.path.join(outputs_path, item) for item in files 
                  if os.path.isdir(os.path.join(outputs_path, item)) and item != "sample"]
        folders.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return folders
    except:
        return []

def get_samples(lora_name):
    output_name = slugify(lora_name)
    try:
        samples_path = resolve_path_without_quotes(f"outputs/{output_name}/sample")
        files = [os.path.join(samples_path, file) for file in os.listdir(samples_path)]
        files.sort(key=lambda file: os.path.getctime(file), reverse=True)
        return files
    except:
        return []

def start_training(base_model, lora_name, train_script, train_config, sample_prompts):
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists("outputs"):
        os.makedirs("outputs", exist_ok=True)
    
    output_name = slugify(lora_name)
    output_dir = resolve_path_without_quotes(f"outputs/{output_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    download(base_model)
    
    file_type = "sh" if sys.platform != "win32" else "bat"
    sh_filepath = resolve_path_without_quotes(f"outputs/{output_name}/train.{file_type}")
    
    with open(sh_filepath, 'w', encoding="utf-8") as file:
        file.write(train_script)
    gr.Info(f"Generated train script", duration=3)
    
    dataset_path = resolve_path_without_quotes(f"outputs/{output_name}/dataset.toml")
    with open(dataset_path, 'w', encoding="utf-8") as file:
        file.write(train_config)
    gr.Info(f"Generated dataset.toml", duration=3)
    
    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, 'w', encoding='utf-8') as file:
        file.write(sample_prompts)
    gr.Info(f"Generated sample_prompts.txt", duration=3)
    
    command = f"bash \"{sh_filepath}\"" if sys.platform != "win32" else sh_filepath
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['LOG_LEVEL'] = 'DEBUG'
    runner = LogsViewRunner()
    cwd = os.path.dirname(os.path.abspath(__file__))
    gr.Info(f"Started training", duration=3)
    yield from runner.run_command([command], cwd=cwd)
    
    config = toml.loads(train_config)
    concept_sentence = config['datasets'][0]['subsets'][0]['class_tokens']
    sample_prompts_path = resolve_path_without_quotes(f"outputs/{output_name}/sample_prompts.txt")
    with open(sample_prompts_path, "r", encoding="utf-8") as f:
        sample_prompts_list = [line.strip() for line in f.readlines() 
                             if len(line.strip()) > 0 and line[0] != "#"]
    md = readme(base_model, lora_name, concept_sentence, sample_prompts_list)
    readme_path = resolve_path_without_quotes(f"outputs/{output_name}/README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md)
    
    gr.Info(f"Training Complete. Check the outputs folder for the LoRA files.", duration=None)

def update(base_model, lora_name, resolution, seed, workers, class_tokens, learning_rate,
           network_dim, max_train_epochs, save_every_n_epochs, timestep_sampling,
           guidance_scale, vram, num_repeats, sample_prompts, sample_every_n_steps,
           *advanced_components):
    output_name = slugify(lora_name)
    dataset_folder = f"datasets/{output_name}"
    sh = gen_sh(base_model, output_name, resolution, seed, workers, learning_rate,
                network_dim, max_train_epochs, save_every_n_epochs, timestep_sampling,
                guidance_scale, vram, sample_prompts, sample_every_n_steps,
                *advanced_components)
    toml = gen_toml(dataset_folder, resolution, class_tokens, num_repeats)
    return gr.update(value=sh), gr.update(value=toml), dataset_folder

def loaded():
    global current_account
    current_account = account_hf()
    if current_account:
        return (gr.update(value=current_account["token"]), gr.update(visible=False),
                gr.update(visible=True), gr.update(value=current_account["account"], visible=True))
    else:
        return gr.update(value=""), gr.update(visible=True), gr.update(visible=False), gr.update(value="", visible=False)

def update_sample(concept_sentence):
    return gr.update(value=concept_sentence)

def refresh_publish_tab():
    return gr.Dropdown(label="Trained LoRAs", choices=get_loras())

def init_advanced():
    basic_args = {
        'pretrained_model_name_or_path', 'clip_l', 't5xxl', 'ae', 'cache_latents_to_disk',
        'save_model_as', 'sdpa', 'persistent_data_loader_workers', 'max_data_loader_n_workers',
        'seed', 'gradient_checkpointing', 'mixed_precision', 'save_precision', 'network_module',
        'network_dim', 'learning_rate', 'cache_text_encoder_outputs', 'cache_text_encoder_outputs_to_disk',
        'fp8_base', 'highvram', 'max_train_epochs', 'save_every_n_epochs', 'dataset_config', 'output_dir',
        'output_name', 'timestep_sampling', 'discrete_flow_shift', 'model_prediction_type', 'guidance_scale',
        'loss_type', 'optimizer_type', 'optimizer_args', 'lr_scheduler', 'sample_prompts', 'sample_every_n_steps',
        'max_grad_norm', 'split_mode', 'network_args'
    }

    parser = train_network.setup_parser()
    flux_train_utils.add_flux_train_arguments(parser)
    args_info = {}
    for action in parser._actions:
        if action.dest != 'help':
            args_info[action.dest] = {
                "action": action.option_strings, "type": action.type,
                "help": action.help, "default": action.default, "required": action.required
            }
    
    temp = [{'key': k, 'action': v} for k, v in args_info.items()]
    temp.sort(key=lambda x: x['key'])
    
    advanced_component_ids = []
    advanced_components = []
    
    for item in temp:
        key = item['key']
        action = item['action']
        if key not in basic_args:
            action_type = str(action['type'])
            component = None
            with gr.Column(min_width=300):
                if action_type == "None":
                    component = gr.Checkbox()
                else:
                    component = gr.Textbox(value="")
                if component:
                    component.interactive = True
                    component.elem_id = action['action'][0] if action['action'] else key
                    component.label = component.elem_id
                    component.elem_classes = ["advanced"]
                    if action['help']:
                        component.info = action['help']
                advanced_components.append(component)
                advanced_component_ids.append(component.elem_id)
    
    return advanced_components, advanced_component_ids

theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)

css = """
@keyframes rotate { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
#advanced_options .advanced:nth-child(even) { background: rgba(0,0,100,0.04) !important; }
h1{font-family: georgia; font-style: italic; font-weight: bold; font-size: 30px; letter-spacing: -1px;}
h3{margin-top: 0}
.tabitem{border: 0px}
.group_padding{}
nav{position: fixed; top: 0; left: 0; right: 0; z-index: 1000; text-align: center; padding: 10px; box-sizing: border-box; display: flex; align-items: center; backdrop-filter: blur(10px);}
nav button { background: none; color: firebrick; font-weight: bold; border: 2px solid firebrick; padding: 5px 10px; border-radius: 5px; font-size: 14px;}
nav img { height: 40px; width: 40px; border-radius: 40px;}
nav img.rotate { animation: rotate 2s linear infinite;}
.flexible { flex-grow: 1;}
.tast-details { margin: 10px 0 !important;}
.toast-wrap { bottom: var(--size-4) !important; top: auto !important; border: none !important; backdrop-filter: blur(10px);}
.toast-title, .toast-text, .toast-icon, .toast-close { color: black !important; font-size: 14px;}
.toast-body { border: none !important;}
#terminal { box-shadow: none !important; margin-bottom: 25px; background: rgba(0,0,0,0.03);}
#terminal .generating { border: none !important;}
#terminal label { position: absolute !important;}
.tabs { margin-top: 50px;}
.hidden { display: none !important;}
.codemirror-wrapper .cm-line { font-size: 12px !important;}
label { font-weight: bold !important;}
#start_training.clicked { background: silver; color: black;}
"""

js = """
function() {
    let autoscroll = document.querySelector("#autoscroll")
    if (window.iidxx) { window.clearInterval(window.iidxx); }
    window.iidxx = window.setInterval(function() {
        let text = document.querySelector(".codemirror-wrapper .cm-line");
        let img = document.querySelector("#logo");
        if (text && text.innerText.trim().length > 0) {
            autoscroll.classList.remove("hidden");
            if (autoscroll.classList.contains("on")) {
                autoscroll.textContent = "Autoscroll ON";
                window.scrollTo(0, document.body.scrollHeight, { behavior: "smooth" });
                img.classList.add("rotate");
            } else {
                autoscroll.textContent = "Autoscroll OFF";
                img.classList.remove("rotate");
            }
        }
    }, 500);
    autoscroll.addEventListener("click", (e) => { autoscroll.classList.toggle("on"); });

    function debounce(fn, delay) {
        let timeoutId;
        return function(...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn(...args), delay);
        };
    }

    const debouncedClick = debounce(() => document.querySelector("#refresh").click(), 1000);
    document.addEventListener("input", debouncedClick);

    document.querySelector("#start_training").addEventListener("click", (e) => {
        e.target.classList.add("clicked");
        e.target.innerHTML = "Training...";
    });
}
"""

current_account = account_hf()

with gr.Blocks(elem_id="app", theme=theme, css=css, fill_width=True) as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Gym"):
            output_components = []
            caption_list = []
            
            with gr.Row():
                gr.HTML("""<nav>
            <img id='logo' src='/file=icon.png' width='80' height='80'>
            <div class='flexible'></div>
            <button id='autoscroll' class='on hidden'></button>
        </nav>""")
            
            with gr.Row(elem_id='container'):
                with gr.Column():
                    gr.Markdown("""# Step 1. LoRA Info
        <p style="margin-top:0">Configure your LoRA train settings.</p>""", elem_classes="group_padding")
                    
                    lora_name = gr.Textbox(
                        label="The name of your LoRA",
                        info="This has to be a unique name",
                        placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
                    )
                    concept_sentence = gr.Textbox(
                        elem_id="--concept_sentence",
                        label="Trigger word/sentence",
                        info="Trigger word or sentence to be used",
                        placeholder="uncommon word like p3rs0n or trtcrd, or sentence like 'in the style of CNSTLL'",
                        interactive=True,
                    )
                    model_names = list(models.keys())
                    base_model = gr.Dropdown(label="Base model (edit models.yaml to add more)", choices=model_names, value=model_names[0])
                    vram = gr.Radio(["20G", "16G", "12G"], value="20G", label="VRAM", interactive=True)
                    num_repeats = gr.Number(value=10, precision=0, label="Repeat trains per image", interactive=True)
                    max_train_epochs = gr.Number(label="Max Train Epochs", value=16, interactive=True)
                    total_steps = gr.Number(0, interactive=False, label="Expected training steps")
                    sample_prompts = gr.Textbox("", lines=5, label="Sample Image Prompts (Separate with new lines)", interactive=True)
                    sample_every_n_steps = gr.Number(0, precision=0, label="Sample Image Every N Steps", interactive=True)
                    resolution = gr.Number(value=512, precision=0, label="Resize dataset images")
                
                with gr.Column():
                    gr.Markdown("""# Step 2. Dataset
        <p style="margin-top:0">Make sure the captions include the trigger word.</p>""", elem_classes="group_padding")
                    
                    images = gr.File(
                        file_types=["image", ".txt"],
                        label="Upload your images",
                        file_count="multiple",
                        interactive=True,
                        visible=True,
                        scale=1,
                    )
                    
                    with gr.Group(visible=False) as captioning_area:
                        do_captioning = gr.Button("Add AI captions")
                        
                        for i in range(1, MAX_IMAGES + 1):
                            with gr.Row(visible=False) as row:
                                image = gr.Image(
                                    type="filepath", width=111, height=111, min_width=111,
                                    interactive=False, scale=2, show_label=False,
                                    show_share_button=False, show_download_button=False
                                )
                                caption = gr.Textbox(label=f"Caption {i}", scale=15, interactive=True)
                            
                            # Ajouter les 3 composants (row, image, caption) à la liste
                            output_components.extend([row, image, caption])
                            caption_list.append(caption)
                    
                    # Les composants refresh et start sont déjà créés plus bas
                    # On les ajoutera à output_components après leur création
                
                with gr.Column():
                    gr.Markdown("""# Step 3. Train
        <p style="margin-top:0">Press start to start training.</p>""", elem_classes="group_padding")
                    
                    refresh = gr.Button("Refresh", elem_id="refresh", visible=False)
                    start = gr.Button("Start training", visible=False, elem_id="start_training")
                    
                    # Ajouter refresh et start à la fin de output_components
                    output_components.append(refresh)
                    output_components.append(start)
                    
                    train_script = gr.Textbox(label="Train script", max_lines=100, interactive=True)
                    train_config = gr.Textbox(label="Train config", max_lines=100, interactive=True)
            
            with gr.Accordion("Advanced options", elem_id='advanced_options', open=False):
                with gr.Row():
                    seed = gr.Number(label="--seed", info="Seed", value=42, interactive=True)
                    workers = gr.Number(label="--max_data_loader_n_workers", info="Number of Workers", value=2, interactive=True)
                    learning_rate = gr.Textbox(label="--learning_rate", info="Learning Rate", value="8e-4", interactive=True)
                    save_every_n_epochs = gr.Number(label="--save_every_n_epochs", info="Save every N epochs", value=4, interactive=True)
                    guidance_scale = gr.Number(label="--guidance_scale", info="Guidance Scale", value=1.0, interactive=True)
                    timestep_sampling = gr.Textbox(label="--timestep_sampling", info="Timestep Sampling", value="shift", interactive=True)
                    network_dim = gr.Number(label="--network_dim", info="LoRA Rank", value=4, minimum=4, maximum=128, step=4, interactive=True)
                    advanced_components, advanced_component_ids = init_advanced()
            
            with gr.Row():
                terminal = LogsView(label="Train log", elem_id="terminal")
            
            with gr.Row():
                gallery = gr.Gallery(get_samples, inputs=[lora_name], label="Samples", every=10, columns=6)

        with gr.TabItem("Publish") as publish_tab:
            hf_token = gr.Textbox(label="Huggingface Token")
            hf_login = gr.Button("Login")
            hf_logout = gr.Button("Logout")
            
            with gr.Row():
                gr.Markdown("**LoRA**")
                gr.Markdown("**Upload**")
            
            with gr.Row():
                lora_rows = refresh_publish_tab()
                with gr.Column():
                    with gr.Row():
                        repo_owner = gr.Textbox(label="Account", interactive=False)
                        repo_name = gr.Textbox(label="Repository Name")
                    repo_visibility = gr.Textbox(label="Repository Visibility ('public' or 'private')", value="public")
                    upload_button = gr.Button("Upload to HuggingFace")
                    upload_button.click(fn=upload_hf, inputs=[base_model, lora_rows, repo_owner, repo_name, repo_visibility, hf_token])
            
            hf_login.click(fn=login_hf, inputs=[hf_token], outputs=[hf_token, hf_login, hf_logout, repo_owner])
            hf_logout.click(fn=logout_hf, outputs=[hf_token, hf_login, hf_logout, repo_owner])

    publish_tab.select(refresh_publish_tab, outputs=lora_rows)
    lora_rows.select(fn=set_repo, inputs=[lora_rows], outputs=[repo_name])

    dataset_folder = gr.State()
    
    listeners = [
        base_model, lora_name, resolution, seed, workers, concept_sentence,
        learning_rate, network_dim, max_train_epochs, save_every_n_epochs,
        timestep_sampling, guidance_scale, vram, num_repeats, sample_prompts,
        sample_every_n_steps, *advanced_components
    ]
    
    # IMPORTANT: Capturer les IDs et valeurs AVANT de démarrer l'interface
    advanced_component_ids = [x.elem_id for x in advanced_components]
    original_advanced_component_values = [comp.value for comp in advanced_components]

    # Les événements doivent être créés APRES que tous les composants sont définis
    images.upload(load_captioning, inputs=[images, concept_sentence], outputs=output_components)
    images.delete(load_captioning, inputs=[images, concept_sentence], outputs=output_components)
    images.clear(hide_captioning, outputs=[captioning_area, start])
    
    max_train_epochs.change(fn=update_total_steps, inputs=[max_train_epochs, num_repeats, images], outputs=[total_steps])
    num_repeats.change(fn=update_total_steps, inputs=[max_train_epochs, num_repeats, images], outputs=[total_steps])
    images.upload(fn=update_total_steps, inputs=[max_train_epochs, num_repeats, images], outputs=[total_steps])
    images.delete(fn=update_total_steps, inputs=[max_train_epochs, num_repeats, images], outputs=[total_steps])
    images.clear(fn=update_total_steps, inputs=[max_train_epochs, num_repeats, images], outputs=[total_steps])
    
    concept_sentence.change(fn=update_sample, inputs=[concept_sentence], outputs=sample_prompts)
    
    start.click(
        fn=create_dataset,
        inputs=[dataset_folder, resolution, images] + caption_list,
        outputs=dataset_folder
    ).then(
        fn=start_training,
        inputs=[base_model, lora_name, train_script, train_config, sample_prompts],
        outputs=terminal
    )
    
    do_captioning.click(
        fn=run_captioning,
        inputs=[images, concept_sentence] + caption_list,
        outputs=caption_list
    )
    
    demo.load(fn=loaded, js=js, outputs=[hf_token, hf_login, hf_logout, repo_owner])

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    demo.launch(debug=True, show_error=True, allowed_paths=[cwd])
