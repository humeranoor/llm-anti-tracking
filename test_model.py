import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Import tqdm for progress bars


def evaluate_model(model_path, test_data_path):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        use_safetensors=True  # Explicitly enable safetensors
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create classifier
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1  # Use CPU (-1) or specify GPU (0, 1, etc.)
    )
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    print(f"Loaded {len(test_df)} test samples")
    
    # Make predictions on all test URLs
    predictions = []
    probabilities = []
    
    #for url in test_df['url']:
    for i, url in enumerate(tqdm(test_df['url'], desc="Processing URLs", unit="url")):

        try:
            result = classifier(url)
            # Get prediction (LABEL_0 = non-tracker, LABEL_1 = tracker)
            label = 1 if result[0]['label'] == 'LABEL_1' else 0
            predictions.append(label)
            # Store probability of positive class (tracker)
            if result[0]['label'] == 'LABEL_1':
                probabilities.append(result[0]['score'])
            else:
                probabilities.append(1 - result[0]['score'])

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_df)} URLs")
 
        except Exception as e:
            print(f"Error processing URL: {url}")
            print(f"Error details: {e}")
            # Assign default values on error
            predictions.append(0)
            probabilities.append(0.0)
    
    # Convert to numpy arrays
    true_labels = test_df['is_tracker'].values
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Calculate metrics
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions)
    rec = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, probabilities)
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Print results
    print("\n===== Model Evaluation Results =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"True Positives (TP): {tp} - Correctly identified trackers")
    print(f"False Positives (FP): {fp} - Non-trackers incorrectly flagged as trackers")
    print(f"True Negatives (TN): {tn} - Correctly identified non-trackers")
    print(f"False Negatives (FN): {fn} - Trackers missed")
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nSpecificity: {specificity:.4f} - Ability to correctly identify non-trackers")
    
    # Class distribution
    pos_rate = np.mean(true_labels)
    print(f"\nClass Distribution:")
    print(f"Trackers: {np.sum(true_labels)} ({pos_rate:.2%})")
    print(f"Non-trackers: {len(true_labels) - np.sum(true_labels)} ({1-pos_rate:.2%})")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Tracker', 'Tracker'],
                yticklabels=['Non-Tracker', 'Tracker'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix visualization saved as 'confusion_matrix.png'")
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

def test_individual_urls():
    # Test specific URLs
    model = AutoModelForSequenceClassification.from_pretrained(
        "./final_model",
        use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained("./final_model")
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )
    
    # Test URLs
    test_urls = [
        "https://cdn.shopify.com/zkesovc/vgvtp.js",  # 0
        "https://netflix.com/idfl/wsmi/gkybuiu/kcnikt",  # 0
        "https://cdn.shopify.com/ulwt/tamksdg/fotaubzv.woff2",  # 0
        "https://wikipedia.org/mhubmuxn/doycsh/ourssv?utm_473=zvflkp&ref=805=keliur&utm_666=fybpbe/ads?/",  # 1
        "https://youtube.com/waxwvp?ref=223=dqjrhh/ads?/"  # 1
    ]
    
    print("\n===== Individual URL Tests =====")
    for url in test_urls:
        result = classifier(url)
        predicted_label = "Tracker" if result[0]['label'] == 'LABEL_1' else "Non-Tracker"
        print(f"URL: {url}")
        print(f"Prediction: {predicted_label} ({result[0]['score']:.2%})\n")

if __name__ == "__main__":
    # Run individual URL tests
    test_individual_urls()

    # Run evaluation on test dataset
    metrics = evaluate_model(
        model_path="./final_model",
        test_data_path="./data/testing_data.csv"
    )
    
