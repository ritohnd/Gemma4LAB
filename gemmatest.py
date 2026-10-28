import os
# Forces PyTorch to handle memory segments cleanly and prevent fragmentation cracks
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import transformers
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig

# Squelch the aggressive internal 6.43 GB pre-allocation warmup spike
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

MODEL_ID = "google/gemma-4-E4B-it"

print("Loading processor parameters from Hub...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure native 4-bit compression variables
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True  # Safely spills ancillary parameters to system RAM
)

# Custom device map strategy: Pins the heavy 150M vision and 300M audio encoders 
# permanently to your system CPU, allowing the text transformer layers to fit 
# completely into your 4-bit VRAM footprint.
custom_device_map = {
    "model.vision_encoder": "cpu",
    "model.audio_encoder": "cpu",
    "model": 0,
    "lm_head": 0
}

print("Loading model via layered block-by-block GPU quantization...")
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map=custom_device_map,
    low_cpu_mem_usage=False  # Crucial: Keeps tensors real to avoid .item() meta crashes
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

print("Compiling conversation tokens...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False
).to("cuda:0")

input_len = inputs["input_ids"].shape[-1]

print("Generating response text...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=64
    )

# Slice and decode the output tokens generated past the initial prompt length
response = processor.decode(outputs[input_len:], skip_special_tokens=True)

print("\n--- Model Output ---")
print(response)
