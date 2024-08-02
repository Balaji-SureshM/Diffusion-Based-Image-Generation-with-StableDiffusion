from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image

model_id = "stabilityai/stable-diffusion-2"

try:
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")


    prompt = "playing cricket in a street"
    print(f"Generating image for prompt: '{prompt}'")

    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
    
    image.save("astronaut_rides_horse.png")

except Exception as e:
    print(f"An error occurred: {e}")
