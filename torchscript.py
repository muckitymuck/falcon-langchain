import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Replace 'model_name' with the name of the Hugging Face model you want to use
model_name = 'tiiuae/falcon-7b'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Save the model in TorchScript format
torch.jit.save(torch.jit.script(model), "torchscript_model.pt")
