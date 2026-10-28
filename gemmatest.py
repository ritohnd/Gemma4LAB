import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import transformers
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

# --- MONKEY PATCH: Squelch the aggressive pre-allocation spike ---
def bypass_warmup(*args, **kwargs):
    print(" Bypassing internal caching_allocator_warmup to save VRAM...")
    return None

# Overwrites the internal function so it doesn't request an extra 6.43 GB
transformers.modeling_utils.caching_allocator_warmup = bypass_warmup
# -----------------------------------------------------------------

MODEL_ID = "google/gemma-4-E4B-it"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

# Custom device map forcing non-text peripheral layers to system RAM 
# so your 8 GB VRAM has plenty of space for generation.
custom_device_map = {
    "model.vision_encoder": "cpu",
    "model.audio_encoder": "cpu",
    "model": "cuda:0",
    "lm_head": "cuda:0"
}

print("Loading model directly to host memory...")
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map=custom_device_map,
    low_cpu_mem_usage=False
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

print("Processing input template...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False
).to("cuda:0")

input_len = inputs["input_ids"].shape[-1]

print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=64
    )

response = processor.decode(outputs[input_len:], skip_special_tokens=True)

print("\n--- Model Output ---")
print(response)
