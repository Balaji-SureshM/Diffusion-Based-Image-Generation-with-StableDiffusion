# Diffusion-Based Image Generation with StableDiffusion

This repository contains code for generating images using diffusion models, specifically leveraging the StableDiffusion framework. The code includes examples and utilities for setting up and running image generation tasks with prompts. This repository provides a streamlined approach to exploring and experimenting with state-of-the-art diffusion models for creative image synthesis tasks.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction
StableDiffusion is a powerful framework for image generation using diffusion models. This repository demonstrates how to use the StableDiffusionPipeline along with a custom scheduler to generate high-quality images from textual prompts.

## Installation
To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Diffusion-Based-Image-Generation-with-StableDiffusion.git
cd diffusion-image-generation
pip install -r requirements.txt
```

##  Usage
Here is an example of how to use the provided code to generate an image:
```bash
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from PIL import Image

model_id = "stabilityai/stable-diffusion-2"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "cricket in the street"
print(f"Generating image for prompt: '{prompt}'")

with torch.autocast("cuda"):
    image = pipe(prompt).images[0]

image.save("cricket_in_the_street.png")
```

##Examples
Here are a few example prompts and their corresponding generated images:

Prompt: "A serene landscape with mountains and a river"

Prompt: "A futuristic cityscape at night"

##Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

##License
This project is licensed under the MIT License. See the LICENSE file for more details.





