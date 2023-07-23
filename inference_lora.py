import os
import torch
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import random
import numpy as np

model_path = "checkpoint/stable-diffusion-v1-4"
lora_dir = "checkpoint/refl_lora"
model_name = "refl_lora"
seed = 1234
device = 'cuda'
prompt = "A painting of a girl walking in a hallway and suddenly finds a giant sunflower on the floor blocking her way."

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)
if not os.path.isdir(f"result/{model_name}"):
    os.makedirs(f"result/{model_name}", exist_ok=True)

# create pipeline
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)
pipe.to("cuda")

# load attention processors
pipe.unet.load_attn_procs(lora_dir, weight_name="pytorch_lora_weights.bin")

# run inference
generator = torch.Generator(device=device)
generator = generator.manual_seed(seed)
images = pipe(prompt, num_images_per_prompt=10, generator=generator).images
for idx in range(len(images)):
    images[idx].save(f"result/{model_name}/{idx}.png")