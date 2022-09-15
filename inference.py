from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "textual_inversion_spyro-dragon"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

prompt = "A cat in the style of <spyro-dragon>."

with autocast("cuda"):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("cat-dragon.png")