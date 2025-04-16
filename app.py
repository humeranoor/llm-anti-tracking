# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS to allow browser requests

# Load the model and tokenizer
MODEL_PATH = "./final_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # Set to evaluation mode

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get('url', '')
    
    # Tokenize the URL
    inputs = tokenizer(url, return_tensors="pt", truncation=True, max_length=64)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get the probability for the "tracker" class (index 1)
    tracker_prob = probs[0][1].item()
    
    return jsonify({
        "is_tracker": bool(tracker_prob > 0.5),
        "confidence": float(tracker_prob)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)