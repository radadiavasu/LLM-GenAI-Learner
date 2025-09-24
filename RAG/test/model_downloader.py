from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-3B-Instruct"
local_dir = "E:/Llama-3.2-3B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)

# Load model (no 'device_type' arg, transformers handles this internally)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=local_dir,
    torch_dtype="auto",
    device_map="auto"   # GPU if available, CPU fallback
)

# Quick test
inputs = tokenizer("Hello! Can you explain black holes simply?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))