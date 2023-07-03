import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Replace 'model_name' with the name of the Hugging Face model you want to use
model_name = 'model_name'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model in TorchScript format
torch.jit.save(torch.jit.script(model), "torchscript_model.pt")
