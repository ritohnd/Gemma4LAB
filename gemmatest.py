import torch
from transformers import pipeline, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E4B-it"

print("Setting up 4-bit compression config...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
    # Removed CPU offload to prevent meta-tensor generation errors
)

print("Initializing Gemma 4 pipeline directly on CUDA...")
# The pipeline handles the underlying model.to("cuda") state cleanly without meta device splits
pipe = pipeline(
    task="any-to-any",
    model=MODEL_ID,
    model_kwargs={"quantization_config": quantization_config, "low_cpu_mem_usage": True},
    device_map="cuda:0"  # Forces explicit target mapping to your real GPU
)

# Set up chat query structured input
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]

print("Generating response...")
# Pipeline automatically applies chat templates and extracts text variables natively
outputs = pipe(messages, max_new_tokens=1024, generate_kwargs={"enable_thinking": False})

print("\n--- Model Output ---")
print(outputs[0]["generated_text"])
