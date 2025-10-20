from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import math

# ---------------------------
# Pretrained Models
# ---------------------------

# 1. Domain Routing Model (Text Classification)
domain_classifier = pipeline(
    "text-classification",
    model="your-fine-tuned-domain-model",  # Replace with a fine-tuned domain classifier
    tokenizer="your-fine-tuned-domain-model"
)

# 2. Embedding Model for FAISS
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# FAISS Setup
# ---------------------------

# Initialize empty index (L2 distance)
embedding_dim = 384  # all-MiniLM-L6-v2 output dim
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Keep a mapping of index -> news text
news_history = []

# ---------------------------
# Agent Functions
# ---------------------------

def domain_routing_agent(news_text):
    """
    Classify news domain and assign a probability of being fake
    """
    result = domain_classifier(news_text)[0]
    domain = result['label']
    confidence = result['score']

    domain_fake_prob_map = {
        'Politics': 0.4,
        'Sports': 0.2,
        'Health': 0.25,
        'Entertainment': 0.3,
        'General': 0.3
    }
    prob_fake = domain_fake_prob_map.get(domain, 0.3)
    return prob_fake, confidence

# Example placeholder agents
def prime_actor_agent(news_text):
    return 0.4, 0.7

def official_source_agent(news_text):
    return 0.2, 0.9

def propaganda_agent(news_text):
    return 0.5, 0.6

def rag_inconsistency_agent(news_text):
    """
    Retrieve similar historical news from FAISS and compare
    """
    if len(news_history) == 0:
        return 0.3, 0.8  # no history yet

    query_vec = embed_model.encode([news_text]).astype("float32")
    k = min(3, len(news_history))
    distances, indices = faiss_index.search(query_vec, k)

    # Simple heuristic: if very similar news exists, reduce probability of being fake
    avg_distance = distances.mean()
    prob_fake = 0.3 if avg_distance < 0.5 else 0.5
    confidence = 0.8
    return prob_fake, confidence

# ---------------------------
# Orchestration Layer
# ---------------------------

def orchestration_layer(agent_outputs):
    log_odds_sum = 0
    total_confidence = 0
    for prob, conf in agent_outputs:
        log_odds = math.log(prob / (1 - prob))
        log_odds_sum += log_odds * conf
        total_confidence += conf
    avg_log_odds = log_odds_sum / total_confidence
    final_prob = 1 / (1 + math.exp(-avg_log_odds))
    verdict = "Fake" if final_prob >= 0.5 else "Real"
    return verdict, final_prob

# ---------------------------
# Pipeline Function
# ---------------------------

def fake_news_pipeline(news_text):
    # Step 1: Generate embeddings and store in FAISS
    embedding_vec = embed_model.encode([news_text]).astype("float32")
    faiss_index.add(embedding_vec)
    news_history.append(news_text)

    # Step 2: Run all agents
    agents = [
        domain_routing_agent,
        prime_actor_agent,
        official_source_agent,
        propaganda_agent,
        rag_inconsistency_agent
    ]
    agent_outputs = [agent(news_text) for agent in agents]

    # Step 3: Orchestrate
    verdict, final_prob = orchestration_layer(agent_outputs)
    return verdict, final_prob

# ---------------------------
# Example Usage
# ---------------------------

news_example = "Government announces new health guidelines for COVID-19 vaccination."

verdict, probability = fake_news_pipeline(news_example)
print(f"Verdict: {verdict}, Probability of Fake: {probability:.2f}")
