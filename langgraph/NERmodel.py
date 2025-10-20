# ner_named_entities.py
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

DEFAULT_MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"

def load_ner_model(local_path=None):
    """
    Load NER model and tokenizer.
    If local_path exists and is valid, load from disk.
    Otherwise, download pretrained model from Hugging Face and save locally.
    """
    if local_path:
        local_dir = Path(local_path)
        if local_dir.exists() and (local_dir / "pytorch_model.bin").exists() and (local_dir / "tokenizer.json").exists():
            print(f"Loading model and tokenizer from local path: {local_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_dir)
            model = AutoModelForTokenClassification.from_pretrained(local_dir)
        else:
            print(f"Local path incomplete or missing. Downloading pretrained model: {DEFAULT_MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
            model = AutoModelForTokenClassification.from_pretrained(DEFAULT_MODEL_NAME)
            local_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(local_dir)
            model.save_pretrained(local_dir)
    else:
        print(f"Loading pretrained model from Hugging Face Hub: {DEFAULT_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(DEFAULT_MODEL_NAME)

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_pipeline

def extract_named_entities(entities):
    """
    Extract only meaningful named entities (PER, ORG, MISC) and merge consecutive tokens.
    Apply heuristic for common titles like 'Prime Minister'.
    """
    merged_entities = []
    current_entity = None

    for e in entities:
        if e['entity_group'] in ['PER', 'ORG', 'MISC']:
            if current_entity and e['start'] <= current_entity['end'] + 2:
                current_entity['word'] += " " + e['word']
                current_entity['end'] = e['end']
                current_entity['score'] = max(current_entity['score'], e['score'])
            else:
                if current_entity:
                    merged_entities.append(current_entity)
                current_entity = e.copy()
        else:
            if current_entity:
                merged_entities.append(current_entity)
                current_entity = None

    if current_entity:
        merged_entities.append(current_entity)

    # Apply heuristic for titles commonly in news text
    title_keywords = ['Prime Minister', 'Chief Minister', 'President', 'CM']
    final_entities = []
    for ent in merged_entities:
        if any(title.lower() in ent['word'].lower() for title in title_keywords):
            final_entities.append({'word': ent['word'], 'entity_group': 'MISC'})
        else:
            final_entities.append(ent)

    return final_entities

if __name__ == "__main__":
    # Change this to your local model path
    local_model_path = "/Users/laraibnoorien/Desktop/fake_news_detection/NER_model"

    print("Loading NER model...")
    ner_pipeline = load_ner_model(local_model_path)
    print("Model loaded successfully!\n")

    while True:
        text = input("Enter news text (or type 'exit' to quit): ")
        text = text.title()  # simple capitalization fix

        if text.lower() == "exit":
            break
        entities = ner_pipeline(text)
        named_entities = extract_named_entities(entities)

        if named_entities:
            print("\nNamed Entities Detected:")
            for e in named_entities:
                print(f"{e['word']} -> {e['entity_group']}")
        else:
            print("No named entities found.")
        print("-"*50)
