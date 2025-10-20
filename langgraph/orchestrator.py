import math

def orchestration_layer(agent_outputs):
    """
    agent_outputs: list of (probability, confidence) tuples
    Returns: final verdict ("Fake" or "Real") and final probability
    """
    log_odds_sum = 0
    total_confidence = 0
    
    for prob, conf in agent_outputs:
        # Convert probability to log-odds
        log_odds = math.log(prob / (1 - prob))
        # Weight by confidence
        log_odds_sum += log_odds * conf
        total_confidence += conf
    
    # Average log-odds weighted by confidence
    avg_log_odds = log_odds_sum / total_confidence
    final_prob = 1 / (1 + math.exp(-avg_log_odds))
    
    verdict = "Fake" if final_prob >= 0.5 else "Real"
    return verdict, final_prob
