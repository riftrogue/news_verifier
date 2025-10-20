from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Path to your saved model
model_path = "/Users/laraibnoorien/Desktop/fake_news_detection/BERT_agent_6cat"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set evaluation mode
model.eval()

id2label = {
    0:'sports',
    1:'technology',
    2:'entertainment',
    3:'education',
    4:'business',
    5:'political'
}

def classify_news(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    
    # Print probability per category
    print("\nCategory Probabilities:")
    for i, p in enumerate(probs):
        print(f"{id2label[i]}: {p*100:.2f}%")
    
    # Predicted category
    predicted_index = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_index]
    print(f"\nPredicted Category: {predicted_label}")


if __name__ == "__main__":
    while True:
        news_text = input("\nEnter news text (or type 'exit' to quit): ")
        if news_text.lower() == "exit":
            break
        classify_news(news_text)
