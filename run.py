import torch
import torchvision.transforms as T
import numpy as np
from diffusers import FluxPipeline
from diffusers.utils import load_image
from attn_proc.vanilla import VanillaFluxAttnProcessor

pipe = FluxPipeline.from_pretrained("/home/frain/Documents/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

num_inference_steps = 50
prompt=[
"The quick brown fox jumps over the lazy dog"
]
out_width = 1024
out_height = 1024

attn_processor = VanillaFluxAttnProcessor(pipe, prompt, out_width, out_height)

image = pipe(
    prompt=prompt,
    width=out_width,
    height=out_height,
    guidance_scale=3.5,
    num_inference_steps=num_inference_steps,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images

image[0].save(f"out.png")
to_tensor = T.ToTensor()
save_dict = {
    "image": torch.stack([to_tensor(img) for img in image]),
    "attention_map": attn_processor.attention_store,
    "out_width": out_width,
    "out_height": out_height,
}
torch.save(save_dict, "controller_attention_store.pt")
