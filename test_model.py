# evaluate_model.py
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report
import pandas as pd
import torch
import numpy as np
import os

# Configuration
MODEL_DIR = os.path.abspath("./final_model")
TEST_DATA_PATH = os.path.abspath("./test_data_v2.csv")
MAX_LENGTH = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer():
    """Load model with verbose logging"""
    print(f"\nLoading model from: {MODEL_DIR}")
    
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    
    print("Loading model weights...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    
    print("Model loaded successfully!")
    return model, tokenizer

def evaluate_model(model, tokenizer, test_data):
    """Evaluation with progress tracking"""
    print("\nStarting evaluation...")
    print(f"Test samples: {len(test_data)}")
    
    if len(test_data) == 0:
        raise ValueError("Test data is empty!")

    texts = test_data["url"].tolist()
    labels = test_data["is_tracker"].tolist()
    
    predictions = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(texts)//BATCH_SIZE)+1}...")
        
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(batch_preds)
    
    print("\nEvaluation complete!")
    print(classification_report(labels, predictions, target_names=["Non-Tracker", "Tracker"]))

def predict_urls(model, tokenizer, urls):
    """Prediction with verbose output"""
    print(f"\nPredicting for {len(urls)} URLs:")
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"Processing URL {i}/{len(urls)}...")
        inputs = tokenizer(
            url,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        results.append({
            "url": url,
            "is_tracker": bool(probs[0][1] > 0.5),
            "confidence": f"{probs[0][1].item():.1%}"
        })
    
    print("\nPrediction results:")
    for res in results:
        status = "TRACKER" if res["is_tracker"] else "SAFE"
        print(f"{res['url'][:60]}... → {status} ({res['confidence']})")
    return results

if __name__ == "__main__":
    try:
        # Load components
        model, tokenizer = load_model_and_tokenizer()
        
        # Load test data
        print(f"\nLoading test data from: {TEST_DATA_PATH}")
        test_data = pd.read_csv(TEST_DATA_PATH)
        print(f"Test data loaded: {len(test_data)} samples")
        
        # Evaluate
        evaluate_model(model, tokenizer, test_data)
        
        # Sample predictions
        test_urls = [
            "https://adservice.google.com/something",
            "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js",
            "https://tr.some-tracker-domain.net/pixel.gif"
            
            "https://cdn.shopify.com/zkesovc/vgvtp.js", #0
            "https://netflix.com/idfl/wsmi/gkybuiu/kcnikt", #0
            "https://cdn.shopify.com/ulwt/tamksdg/fotaubzv.woff2",  #0
            "https://wikipedia.org/mhubmuxn/doycsh/ourssv?utm_473=zvflkp&ref=805=keliur&utm_666=fybpbe/ads?/",  #1
            "https://youtube.com/waxwvp?ref=223=dqjrhh/ads?/"   #1
    ]
        predict_urls(model, tokenizer, test_urls)
        
    except Exception as e:
        print(f"\nError: {str(e)}")