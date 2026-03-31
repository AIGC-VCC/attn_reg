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

# 1. 提取 Prompt 对应的 Token IDs (FLUX 用 tokenizer_2 处理 512 长度)
text_inputs = pipe.tokenizer_2(
    prompt, 
    padding="max_length", 
    max_length=512, 
    truncation=True, 
    return_tensors="pt"
)
token_ids = text_inputs.input_ids[0].tolist()

# 2. 将 IDs 转换回可读的单词/子词
decoded_tokens = pipe.tokenizer_2.convert_ids_to_tokens(token_ids)

image[0].save(f"out.png")
to_tensor = T.ToTensor()
save_dict = {
    "image": torch.stack([to_tensor(img) for img in image]),
    "attention_map": attn_processor.attention_store.to('cpu'),
    "tokens": decoded_tokens,  # <--- 把文本 token 列表存进去！
    "seq_len": attn_processor.seq_len, # 把长度也存下来，方便切片
    "text_seq_len": attn_processor.text_seq_len,
    "latent_seq_len": attn_processor.latent_seq_len,
    "out_width": out_width,
    "out_height": out_height,
}
torch.save(save_dict, "controller_attention_store.pt")
