import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

SAVE_DIR = '/Users/laraibnoorien/Desktop/fake_news_detection/AGENT1_model'
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR, local_files_only=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, local_files_only=True)


model.eval()

id2label = {0: "politics", 
            1: "sports", 
            2: "entertainment", 
            3: "science", 
            4: "health"}

"""samples = [
    "The government passed a new bill in Parliament today.",
    "The cricket league final was a thrilling match last night.",
]"""

samples= input("Enter a news segment: ")

MAX_LEN = 128

"""for s in samples:
    enc = tokenizer(s, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    
    print("\nTEXT:", s)
    for i, p in enumerate(probs):
        print(f"{id2label[i]:15s}: {p:.3f}")"""

enc = tokenizer(samples, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN).to(device)
with torch.no_grad():
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

print("\nTEXT:", samples)
for i, p in enumerate(probs):
    print(f"{id2label[i]:15s}: {p:.3f}")
