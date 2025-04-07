# train_model.py - Device-Optimized
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"  # Revert to DistilBERT for CPU compatibility
MAX_LENGTH = 64
BATCH_SIZE = 8 if DEVICE == "cpu" else 32
FP16 = False if DEVICE == "cpu" else True

class URLDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]).to(DEVICE),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]).to(DEVICE),
            'labels': torch.tensor(self.labels[idx]).to(DEVICE)
        }

    def __len__(self):
        return len(self.labels)

# Load data
data = pd.read_csv("training_data.csv")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["url"], data["is_tracker"], test_size=0.2, random_state=42
)

# Initialize model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

# Tokenize
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding='max_length',
    max_length=MAX_LENGTH
)
test_encodings = tokenizer(
    test_texts.tolist(),
    truncation=True,
    padding='max_length',
    max_length=MAX_LENGTH
)

# Create datasets
train_dataset = URLDataset(train_encodings, train_labels.tolist())
test_dataset = URLDataset(test_encodings, test_labels.tolist())

# Device-aware training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,
    learning_rate=3e-5,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    fp16=FP16,
    optim="adamw_torch" if DEVICE == "cpu" else "adamw_apex_fused",  # CPU-safe
    report_to="none",
    save_strategy="no"
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print(f"Training on {DEVICE.upper()}...")
trainer.train()

# Save and test
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
print("\nModel saved to './final_anti_tracking_model'")

# Quick test
test_urls = [
    "https://tracker.com/ads.js?uid=abc",
    "https://cdnjs.com/jquery.min.js"
]
inputs = tokenizer(test_urls, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    for url, prob in zip(test_urls, probs):
        print(f"URL: {url[:40]}... → Tracker: {prob[1]:.1%}")


# Final evaluation
print("\nRunning final test...")
test_results = trainer.evaluate()
print(f"Validation Accuracy: {test_results['eval_accuracy']:.2%}")