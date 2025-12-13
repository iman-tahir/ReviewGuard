import torch
from transformers import BertTokenizer, BertForSequenceClassification

class ReviewDetector:
    def __init__(self, model_path='./results/checkpoint-best'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    
    def predict(self, review_text):
        inputs = self.tokenizer(
            review_text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        label = "FAKE" if prediction == 1 else "REAL"
        return {
            'label': label,
            'confidence': confidence,
            'prediction': prediction
        }

# CLI usage
if __name__ == "__main__":
    import sys
    detector = ReviewDetector()
    
    if len(sys.argv) > 1:
        review = " ".join(sys.argv[1:])
        result = detector.predict(review)
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.1%}")
    else:
        print("Usage: python predict.py 'Your review text here'")