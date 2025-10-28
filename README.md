# 🔍 Multi-Agent News Verification System

AI-powered fact-checking system with 6 specialized agents using DuckDuckGo search and Groq LLMs.

## ✨ Features

- 🤖 **6-Agent System**: Domain routing, entity extraction, source verification, bias detection, historical analysis, and fusion
- 🌐 **DuckDuckGo Search**: Real-time web search with privacy focus
- 🔒 **Tor Support**: Optional anonymous web scraping
- 📊 **Smart Scoring**: Mathematical fusion with domain-specific thresholds
- 📱 **Dual Output**: Pretty console display or JSON API

## 🚀 Quick Start

### Setup
```bash
git clone <your-repo>
cd news_verifier
pip install -r requirements.txt
```

### Configure `.env`
```env
GROQ_API_KEY_A1=your_key_here
GROQ_API_KEY_A2=your_key_here
GROQ_API_KEY_A3=your_key_here
GROQ_API_KEY_A4=your_key_here
GROQ_API_KEY_A5=your_key_here
GROQ_API_KEY_A6=your_key_here
USE_TOR=false
```

### Usage
```bash
# Pretty output
python main.py "Your news claim here"

# JSON output  
python main.py --json "Your claim"

# Save results
python main.py --json --save "Your claim"
```

## 📊 Output Example

**Console:**
```
🔍 MULTI-AGENT FACT-CHECKING ANALYSIS
================================================================================
📰 CLAIM: "Mukesh Ambani became India's richest man again"
🎯 VERDICT: ✅ REAL (95% confidence)
📂 DOMAIN: business/finance
👥 ENTITIES: Mukesh Ambani, India
================================================================================
```

**JSON:**
```json
{
  "final_verdict": "Real",
  "confidence": 95,
  "domain": "business/finance", 
  "named_entities": ["Mukesh Ambani", "India"],
  "sources": [{"url": "...", "title": "...", "snippet": "..."}]
}
```

## 🏗️ Architecture

**6 Agents:**
- **A1**: Domain classification 
- **A2**: Entity extraction
- **A3**: Source verification (DuckDuckGo + real-time keywords)
- **A4**: Bias/toxicity detection
- **A5**: Historical context analysis
- **A6**: Intelligent fusion

**Tech Stack:**
- **Search**: DuckDuckGo (ddgs)
- **LLMs**: Groq (Llama-3.3-70b, Llama-3.1-8b)
- **Privacy**: Optional Tor proxy via requests-tor
- **Processing**: Async parallel execution

## 📦 Dependencies

```
python-dotenv
langchain-core>=0.3.0
langchain-groq>=0.2.0
requests>=2.31.0
ddgs>=1.0.0
requests-tor>=1.4
```

## 🔧 Advanced

**Tor Setup:**
1. Install Tor Browser or daemon
2. Set `USE_TOR=true` in `.env`
3. System auto-detects and falls back if unavailable

**Programmatic:**
```python
from llm_agents import MultiAgentFactChecker
import asyncio

async def check_claim():
    checker = MultiAgentFactChecker()
    result = await checker.process_claim("Your claim")
    print(f"Verdict: {result.final_verdict} ({result.confidence}%)")

asyncio.run(check_claim())
```

**Business/Finance Optimization:**
- Uses 0.55 threshold (vs 0.5 default) to reduce false "Fake" verdicts
- Enhanced real-time keyword searches for current financial data
- Treats older trending data as neutral rather than contradictory

---

**Start verifying:** `python main.py "Your news claim here"`