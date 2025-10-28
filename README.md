# 🔍 Multi-Agent News Verification System

AI-powered fact-checking system with 6 specialized agents using DuckDuckGo search and Groq LLMs.

## ✨ Features

- 🤖 **6-Agent System**: Domain routing, entity extraction, source verification, bias detection, historical analysis, and fusion
- 🌐 **DuckDuckGo Search**: Real-time web search with privacy focus
- 🔒 **Tor Support**: Optional anonymous web scraping with zero-configuration setup
- 📊 **Smart Scoring**: Mathematical fusion with domain-specific thresholds
- 📱 **Dual Output**: Pretty console display or JSON API

## 🚀 Complete Setup Guide

### Prerequisites
- **Python 3.8+** (Download from [python.org](https://www.python.org/downloads/))
- **Git** (Download from [git-scm.com](https://git-scm.com/downloads/))
- **Groq API Keys** (Get free keys from [console.groq.com](https://console.groq.com/))

### Step 1: Clone Repository
```bash
git clone https://github.com/riftrogue/news_verifier.git
cd news_verifier
```

### Step 2: Create Virtual Environment

**Windows (PowerShell/CMD):**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Verify activation (should show (.venv) in prompt)
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify activation (should show (.venv) in prompt)
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(ddgs|groq|requests|stem)"
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file (Windows)
echo. > .env

# Create .env file (macOS/Linux) 
touch .env
```

Add your Groq API keys to `.env`:
```env
# Groq API Keys (get 6 keys from https://console.groq.com/)
GROQ_API_KEY_A1=gsk_your_key_1_here
GROQ_API_KEY_A2=gsk_your_key_2_here
GROQ_API_KEY_A3=gsk_your_key_3_here
GROQ_API_KEY_A4=gsk_your_key_4_here
GROQ_API_KEY_A5=gsk_your_key_5_here
GROQ_API_KEY_A6=gsk_your_key_6_here

# Privacy Settings
USE_TOR=false  # Set to 'true' for anonymous searches
```

**🔑 Getting Groq API Keys:**
1. Visit [console.groq.com](https://console.groq.com/)
2. Sign up/login with GitHub or Google
3. Go to API Keys section
4. Create 6 different API keys (one for each agent)
5. Copy each key to your `.env` file

### Step 5: Test Installation
```bash
# Test basic functionality
python main.py "Python is a programming language"

# Test JSON output
python main.py --json "The sky is blue"
```

## 📖 Usage Guide

### Basic Commands

**Pretty Console Output:**
```bash
python main.py "Mukesh Ambani became India's richest man in 2025"
```

**JSON Output (for APIs):**
```bash
python main.py --json "OpenAI released GPT-4o in May 2024"
```

**Save Results to File:**
```bash
python main.py --json --save "Climate change is real"
# Saves to: data/verified_reports.json
```

### Privacy Mode (Tor)

**Enable Anonymous Searches:**
```bash
# Edit .env file
USE_TOR=true

# Run normally - Tor auto-installs on first use!
python main.py "Sensitive political claim here"
```

**How Tor Mode Works:**
- ✅ **Zero Configuration**: Downloads Tor automatically (~15MB first time)
- ✅ **No GUI Required**: Runs headless Tor daemon in background
- ✅ **Anonymous Searches**: All web requests routed through Tor network
- ✅ **Auto Cleanup**: Tor process stops when verification completes

### Command-Line Options

```bash
# Basic usage
python main.py "Your claim here"

# JSON output for integration
python main.py --json "Your claim"

# Save results to data/verified_reports.json
python main.py --save "Your claim"

# JSON output + save file
python main.py --json --save "Your claim"

# Help
python main.py --help
```

## 📊 Output Examples

### Console Output (Pretty Format)
```
🔍 MULTI-AGENT FACT-CHECKING ANALYSIS
================================================================================
📰 CLAIM: OpenAI released GPT-4o in May 2024
--------------------------------------------------------------------------------
🎯 VERDICT & CONFIDENCE
   ✅ FINAL VERDICT: REAL
   📊 CONFIDENCE: [████████████████████] 95%
   📂 DOMAIN: Technology

📋 AGENT ANALYSIS FINDINGS
   1. Domain classified as science/health with high confidence
   2. Found 4 key entities: OpenAI, GPT-4o, May, 2024
   3. Source verification: 2 supporting, 0 contradicting sources
   4. No bias or toxicity indicators detected
   5. Historical context supports the claim

🔗 SOURCES & REFERENCES
   📄 OpenAI Official Blog - GPT-4o Release Announcement
   📄 TechCrunch - OpenAI Unveils GPT-4o Model
   📄 The Verge - GPT-4o Features and Capabilities

✅ Analysis Complete!
```

### JSON Output (API Format)
```json
{
  "claim": "OpenAI released GPT-4o in May 2024",
  "final_verdict": "Real",
  "confidence": 95,
  "domain": "technology",
  "named_entities": ["OpenAI", "GPT-4o", "May", "2024"],
  "agent_scores": {
    "A1_domain": 0.402,
    "A2_entities": 0.300,
    "A3_sources": 0.150,
    "A4_bias": 0.600,
    "A5_historical": 0.400,
    "P_fake": 0.493
  },
  "sources": [
    {
      "url": "https://openai.com/blog/hello-gpt-4o",
      "title": "GPT-4o Release Announcement",
      "snippet": "Today we're announcing GPT-4o..."
    }
  ],
  "processing_time_ms": 12500,
  "timestamp": "2025-10-29T01:35:50Z"
}
```

## 🏗️ System Architecture

### 6-Agent Pipeline
```
Input Claim
     ↓
┌────────────────────────────────────────┐
│  🤖 A1: Domain Classification          │ → science/tech/business/politics/sports
├────────────────────────────────────────┤
│  👥 A2: Entity Extraction              │ → people/orgs/locations/dates
├────────────────────────────────────────┤
│  🔍 A3: Source Verification            │ → web search + credibility analysis
├────────────────────────────────────────┤
│  🎭 A4: Bias & Toxicity Detection      │ → propaganda/manipulation patterns
├────────────────────────────────────────┤
│  📚 A5: Historical Context Analysis    │ → timeline verification + contradictions
├────────────────────────────────────────┤
│  🧠 A6: Intelligent Fusion             │ → mathematical scoring + final verdict
└────────────────────────────────────────┘
     ↓
Final Verdict: REAL/FAKE (confidence %)
```

### Tech Stack
- **🔍 Search Engine**: DuckDuckGo (ddgs package) - Privacy-focused web search
- **🧠 LLMs**: Groq API (Llama-3.3-70b-versatile) - Fast inference
- **🔒 Privacy Layer**: Tor network (optional) - Anonymous web scraping
- **⚡ Processing**: Async/await - Parallel agent execution
- **🎯 Fusion**: Mathematical scoring - Domain-specific thresholds

## 🛠️ Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'ddgs'"**
```bash
# Solution: Reinstall requirements in virtual environment
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**"Groq API rate limit exceeded"**
```bash
# Solution: Wait or use multiple API keys
# Rate limits: 100,000 tokens/day per free account
# Use 6 different Groq accounts for higher limits
```

**"Tor connection failed"**
```bash
# Solution: Disable Tor temporarily
USE_TOR=false

# Or clear Tor cache
rm -rf .tor/  # macOS/Linux  
rmdir /s .tor  # Windows
```

**"UnicodeEncodeError in Windows terminal"**
```bash
# Solution: Use JSON output instead
python main.py --json "Your claim"
```

## 📦 Dependencies

```txt
python-dotenv>=1.0.0    # Environment variable management
langchain-core>=0.3.0   # LLM framework core
langchain-groq>=0.3.0   # Groq API integration
requests>=2.32.0        # HTTP client for web requests
ddgs>=9.6.0            # DuckDuckGo search API
stem>=1.8.0            # Tor control library (for privacy mode)
```

## 🚀 Quick Start Examples

```bash
# 1. Test tech claim
python main.py "ChatGPT was released by OpenAI in 2022"

# 2. Test business claim  
python main.py "Tesla stock hit $1000 per share in 2024"

# 3. Test with privacy mode
echo "USE_TOR=true" >> .env
python main.py "Controversial political statement"

# 4. API integration
python main.py --json "Scientific discovery claim" > result.json
```

---

**🎯 Ready to verify news?** Run: `python main.py "Your news claim here"`

