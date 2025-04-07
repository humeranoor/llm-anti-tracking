import torch
from transformers import AutoModel, AutoTokenizer

print("1. PyTorch device:", torch.device("cpu"))
model = AutoModel.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("2. Model & tokenizer loaded successfully!")

# Test inference
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
print("3. Inference test passed!")