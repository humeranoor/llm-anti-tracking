from flask import Flask, request, jsonify
from flask_cors import CORS  # Handle CORS for extension
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Allow Chrome extension access
classifier = pipeline("text-classification", model="./../final_model")

@app.route('/predict', methods=['POST'])
def predict():
    url = request.json['url']
    result = classifier(url)[0]
    return jsonify({
        "is_tracker": result['label'] == "LABEL_1",
        "confidence": result['score']
    })

if __name__ == '__main__':
    app.run(port=5000)