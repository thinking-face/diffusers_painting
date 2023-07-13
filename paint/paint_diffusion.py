import os
import gradio as gr
from PIL import Image

import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-2"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    image.save("output.png")
    return Image.open("output.png")

if __name__ == '__main__':

    inputs = gr.inputs.Textbox(lines=1, label="Prompt")
    outputs = gr.outputs.Image(type="pil", label="Generated Image")

    title = "Image Generation"
    description = "Generate an image based on the provided prompt."
    examples = [["a photo of cute girl eating ice cream"]]

    gr.Interface(generate_image, inputs, outputs, title=title, description=description, examples=examples, share=True).launch()