# train_model.py - Device-Optimized
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

FP16 = True if DEVICE == "cuda" else False

# Optimizer choice
if DEVICE == "cuda":
    OPTIMIZER = "adamw_apex_fused"
else:
    OPTIMIZER = "adamw_torch"

MODEL_NAME = "distilbert-base-uncased"  
MAX_LENGTH = 64
BATCH_SIZE = 8 if DEVICE == "cpu" else 32

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
print('Reading Ground Truth Data')
data = pd.read_csv("./data/training_data_2804.csv")
print('Split into train and test')

train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["url"], data["is_tracker"], test_size=0.2, random_state=42
)

# Initialize model
print('Getting tokenizer')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print('Getting model')
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

# Save the original pre-trained model
print('saving original model')
model.save_pretrained("./original_distilbert")
print('saving original tokenizer')
tokenizer.save_pretrained("./original_distilbert")

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
print('preparing train set')
train_dataset = URLDataset(train_encodings, train_labels.tolist())
print('preparing test set')
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
    optim=OPTIMIZER,        # ← Here, OPTIMIZER must be set correctly
    report_to="none",
    save_strategy="no"
)




def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    
    # Calculate basic metrics
    acc = accuracy_score(p.label_ids, preds)
    precision = precision_score(p.label_ids, preds)
    recall = recall_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds)
    
    # Calculate ROC AUC (needs predicted probabilities, not class predictions)
    # We use the probability of the positive class (index 1)
    roc_auc = roc_auc_score(p.label_ids, p.predictions[:, 1])
    
    # Get confusion matrix
    cm = confusion_matrix(p.label_ids, preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

print(f"Model is on device: {next(model.parameters()).device}")

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
print('saving finetuned model')
model.save_pretrained("./final_model")
print('saving finetuned tokenizer')
tokenizer.save_pretrained("./final_model")
print("\nModel saved to './final_anti_tracking_model'")

# Quick test
print('doing quicktest of finetuned model')

test_urls = [
            "https://cdn.shopify.com/zkesovc/vgvtp.js", #0
            "https://netflix.com/idfl/wsmi/gkybuiu/kcnikt", #0
            "https://cdn.shopify.com/ulwt/tamksdg/fotaubzv.woff2",  #0
            "https://wikipedia.org/mhubmuxn/doycsh/ourssv?utm_473=zvflkp&ref=805=keliur&utm_666=fybpbe/ads?/",  #1
            "https://youtube.com/waxwvp?ref=223=dqjrhh/ads?/"   #1

#    "https://tracker.com/ads.js?uid=abc",
#    "https://cdnjs.com/jquery.min.js"
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
print(f"Precision: {test_results['eval_precision']:.2%}")
print(f"Recall: {test_results['eval_recall']:.2%}")
print(f"F1 Score: {test_results['eval_f1']:.2%}")
print(f"ROC AUC: {test_results['eval_roc_auc']:.2%}")
print("\nConfusion Matrix:")
print(f"True Positives: {test_results['eval_true_positives']}")
print(f"False Positives: {test_results['eval_false_positives']}")
print(f"True Negatives: {test_results['eval_true_negatives']}")
print(f"False Negatives: {test_results['eval_false_negatives']}")


