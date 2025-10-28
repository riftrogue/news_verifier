import asyncio
import logging
import math
import os
import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from ddgs import DDGS

# Tor proxy support
try:
    from tor_manager import TorManager
    TOR_AVAILABLE = True
except ImportError:
    TOR_AVAILABLE = False

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AgentResult:
    """Standard result format for all agents"""
    agent_id: str
    score: float  # s1, s2, s3, s4, s5
    evidence: List[str]
    sources: List[Dict[str, str]]
    metadata: Dict[str, Any]

@dataclass
class FinalVerdict:
    """Final output format"""
    named_entities: List[str]
    domain: str
    fact_finding: List[str]
    sources: List[Dict[str, str]]
    final_verdict: str  # "Fake" or "Real"
    confidence: int  # percentage

class MultiAgentFactChecker:
    """
    Multi-agent fact-checking system implementing the 6-agent architecture:
    
    A1: Domain Router - Classifies news claims into domains
    A2: Prime-Actor Resolver - Identifies key entities and actors  
    A3: Official-Source Verifier - Checks official sources for verification
    A4: Propaganda and Toxicity Detector - Analyzes for manipulation patterns
    A5: RAG Inconsistency Agent - Checks against historical context
    A6: Orchestrator Agent - Fuses all agent outputs into final verdict
    """
    
    def __init__(self):
        # Initialize search with DuckDuckGo and rate limiting
        self.search_max_results = 10
        self.search_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent searches
        
        # Initialize Tor proxy if available and enabled
        self.use_tor = TOR_AVAILABLE and os.getenv("USE_TOR", "false").lower() == "true"
        self.tor_manager = None
        if self.use_tor:
            try:
                self.tor_manager = TorManager()
                # Auto-install and start Tor
                if self.tor_manager.ensure_tor_installed() and self.tor_manager.start_tor():
                    logger.info("✅ Tor proxy enabled and running")
                else:
                    logger.warning("❌ Failed to start Tor. Falling back to direct connection.")
                    self.use_tor = False
                    self.tor_manager = None
            except Exception as e:
                logger.warning(f"❌ Failed to initialize Tor: {e}. Falling back to direct connection.")
                self.use_tor = False
                self.tor_manager = None
        
        # Initialize LLMs for each agent with specific API keys and optimized models
        self.llms = {
            'A1': ChatGroq(api_key=os.getenv("GROQ_API_KEY_A1"), model="llama-3.3-70b-versatile", temperature=0.1),
            'A2': ChatGroq(api_key=os.getenv("GROQ_API_KEY_A2"), model="llama-3.1-8b-instant", temperature=0.1),
            'A3': ChatGroq(api_key=os.getenv("GROQ_API_KEY_A3"), model="llama-3.3-70b-versatile", temperature=0.1),
            'A4': ChatGroq(api_key=os.getenv("GROQ_API_KEY_A4"), model="llama-3.1-8b-instant", temperature=0.1),
            'A5': ChatGroq(api_key=os.getenv("GROQ_API_KEY_A5"), model="llama-3.3-70b-versatile", temperature=0.1),
            'A6': ChatGroq(api_key=os.getenv("GROQ_API_KEY_A6"), model="llama-3.3-70b-versatile", temperature=0.1)
        }
        
        # Domain-specific fake news priors
        self.domain_priors = {
            'politics': 0.6, 'sports': 0.4, 'entertainment': 0.4,
            'science/health': 0.4, 'business/finance': 0.4, 'general': 0.4
        }
    
    def cleanup(self):
        """Clean up resources, specifically Tor connection."""
        if hasattr(self, 'tor_manager') and self.tor_manager:
            try:
                self.tor_manager.cleanup()
                logger.info("🧹 Tor resources cleaned up")
            except Exception as e:
                logger.warning(f"Warning during Tor cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup is called."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup in destructor

    async def web_search(self, query: str, max_results: Optional[int] = None, timelimit: Optional[str] = None) -> List[Dict[str, str]]:
        """Perform web search using DuckDuckGo with rate limiting and async safety"""
        limit = max_results or self.search_max_results
        results = []
        search_start = asyncio.get_event_loop().time()
        
        async with self.search_semaphore:
            try:
                # Rate limit: 1.5s sleep between searches
                await asyncio.sleep(1.5)
                
                # Use Tor proxy for DDGS if enabled
                if self.use_tor and self.tor_manager:
                    # Get proxy URL from TorManager
                    tor_proxy = self.tor_manager.get_proxy_url()
                    
                    if tor_proxy:
                        logger.info(f"🔒 Using Tor SOCKS proxy: {tor_proxy}")
                    
                    if tor_proxy:
                        try:
                            with DDGS(proxy=tor_proxy, timeout=10) as ddgs:
                                search_results = ddgs.text(query, region="in-en", safesearch="moderate", timelimit=timelimit, max_results=limit)
                                for result in search_results:
                                    results.append({
                                        "url": result.get("href", ""), 
                                        "title": result.get("title", ""), 
                                        "snippet": result.get("body", "")
                                    })
                                
                            search_duration = asyncio.get_event_loop().time() - search_start
                            logger.info(f"🔒 Tor Search '{query[:50]}...' returned {len(results)} results in {search_duration:.2f}s (via SOCKS proxy)")
                        except Exception as tor_error:
                            logger.warning(f"Tor proxy failed for '{query[:50]}...': {tor_error}. Falling back to direct search.")
                            # Fallback to direct search if Tor fails
                            with DDGS() as ddgs:
                                search_results = ddgs.text(query, region="in-en", safesearch="moderate", timelimit=timelimit, max_results=limit)
                                for result in search_results:
                                    results.append({
                                        "url": result.get("href", ""), 
                                        "title": result.get("title", ""), 
                                        "snippet": result.get("body", "")
                                    })
                            search_duration = asyncio.get_event_loop().time() - search_start
                            logger.info(f"Fallback Search '{query[:50]}...' returned {len(results)} results in {search_duration:.2f}s")
                    else:
                        logger.warning("No Tor SOCKS proxy available. Falling back to direct search.")
                        # Fallback to direct search if no Tor proxy found
                        with DDGS() as ddgs:
                            search_results = ddgs.text(query, region="in-en", safesearch="moderate", timelimit=timelimit, max_results=limit)
                            for result in search_results:
                                results.append({
                                    "url": result.get("href", ""), 
                                    "title": result.get("title", ""), 
                                    "snippet": result.get("body", "")
                                })
                        search_duration = asyncio.get_event_loop().time() - search_start
                        logger.info(f"Direct Search '{query[:50]}...' returned {len(results)} results in {search_duration:.2f}s")
                else:
                    with DDGS() as ddgs:
                        search_results = ddgs.text(query, region="in-en", safesearch="moderate", timelimit=timelimit, max_results=limit)
                        for result in search_results:
                            results.append({
                                "url": result.get("href", ""), 
                                "title": result.get("title", ""), 
                                "snippet": result.get("body", "")
                            })
                    search_duration = asyncio.get_event_loop().time() - search_start
                    logger.info(f"Direct Search '{query[:50]}...' returned {len(results)} results in {search_duration:.2f}s")
                        
            except Exception as e:
                search_duration = asyncio.get_event_loop().time() - search_start
                logger.warning(f"DuckDuckGo search error for '{query[:50]}...' after {search_duration:.2f}s: {e}")
        
        return results

    async def fetch_webpage_tor(self, url: str) -> str:
        """Fetch webpage content using Tor if enabled, otherwise direct connection"""
        try:
            if self.use_tor:
                response = self.tor_session.get(url, timeout=10)
                logger.info(f"🔒 Tor fetch: {url[:50]}... status={response.status_code}")
                return response.text
            else:
                # Fallback to direct connection (you could use requests here)
                import requests
                response = requests.get(url, timeout=10)
                logger.info(f"Direct fetch: {url[:50]}... status={response.status_code}")
                return response.text
        except Exception as e:
            logger.warning(f"Web fetch error for {url[:50]}...: {e}")
            return ""

    # ============================================================================
    # AGENT A1: DOMAIN ROUTER  
    # ============================================================================

    async def agent_a1_domain_router(self, claim: str) -> AgentResult:
        """A1: Domain Router - Classifies claim into domains and computes domain-based fake prior"""
        prompt = PromptTemplate.from_template("""
You are a domain classification expert for Indian news claims. 

Classify this news claim into ONE of these domains: politics, sports, entertainment, science/health, business/finance, general

Claim: "{claim}"

Use this format:
DOMAIN: [domain]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]

Examples:
- "Prime Minister announces new policy" → politics
- "Cricket team wins match" → sports  
- "Actor announces new movie" → entertainment
- "WHO announces health guidelines" → science/health
- "Mukesh Ambani net worth increases" → business/finance
- "Stock market crashes" → business/finance
- "Weather update for tomorrow" → general

Respond with ONLY the format above.
        """)
        
        for attempt in range(2):  # Retry logic
            try:
                response = await self.llms['A1'].ainvoke(prompt.format(claim=claim))
                content = response.content
                
                domain_match = re.search(r'DOMAIN:\s*([a-zA-Z/]+)', content, re.IGNORECASE)
                confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content)
                reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.IGNORECASE)
                
                domain = domain_match.group(1).lower() if domain_match else None
                domain_confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
                
                if not domain or domain not in self.domain_priors:
                    if attempt == 0:
                        financial_keywords = ['richest', 'net worth', 'stock', 'market', 'wealth', 'fortune', 'business', 'company', 'billionaire', 'surpassed']
                        if any(keyword in claim.lower() for keyword in financial_keywords):
                            fallback_prompt = f"{prompt.template}\n\nNote: This claim seems related to business/finance based on keywords."
                            response = await self.llms['A1'].ainvoke(fallback_prompt.format(claim=claim))
                            content = response.content
                            domain_match = re.search(r'DOMAIN:\s*([a-zA-Z/]+)', content, re.IGNORECASE)
                            domain = domain_match.group(1).lower() if domain_match else 'business/finance'
                        else:
                            domain = 'general'
                    else:
                        domain = 'general'
                
                if domain not in self.domain_priors:
                    domain = 'general'
                
                mu = 1
                fake_rate = self.domain_priors.get(domain, 0.4)
                N_k = 100
                F_k = int(fake_rate * N_k)
                s1 = (F_k + mu) / (N_k + 2 * mu)
                
                logger.info(f"A1: Classified '{claim[:50]}...' as {domain} with s1={s1:.3f}")
                
                return AgentResult(
                    agent_id="A1", score=s1, 
                    evidence=[f"Domain classified as {domain} with confidence {domain_confidence:.2f}"],
                    sources=[], 
                    metadata={
                        "domain": domain, 
                        "domain_confidence": domain_confidence, 
                        "reasoning": reasoning, 
                        "fake_prior": fake_rate
                    }
                )
            except Exception as e:
                logger.error(f"A1 error (attempt {attempt+1}): {e}")
                if attempt == 1:
                    return AgentResult(
                        "A1", 0.5, 
                        [f"Error in domain classification: {str(e)}"], 
                        [], 
                        {"domain": "general"}
                    )

    # ============================================================================
    # AGENT A2: PRIME-ACTOR RESOLVER  
    # ============================================================================

    async def agent_a2_prime_actor_resolver(self, claim: str) -> AgentResult:
        """A2: Prime-Actor Resolver - Identifies key entities and resolves them"""
        prompt = PromptTemplate.from_template("""
You are an expert in Named Entity Recognition for Indian news claims.

Extract and categorize named entities from this claim, focusing on key actors (people, organizations, locations).

Claim: "{claim}"

For each entity, provide:
1. Entity name
2. Entity type (PERSON, ORG, LOC, MISC)
3. Role/position (if applicable)
4. Reliability assessment (HIGH/MEDIUM/LOW based on how often this entity type appears in fake news)

Format:
ENTITIES:
- [Entity]: [Type] | [Role] | [Reliability]

ACTOR_RISK_SCORE: [0.0-1.0] (higher = more likely to be in fake news)

Examples:
- "Prime Minister": PERSON | Head of Government | HIGH (often misquoted)
- "WHO": ORG | Health Organization | LOW (official source)
- "Bollywood actor": PERSON | Entertainment | MEDIUM (moderate fake news risk)

Focus on PRIMARY actors that would be the subject of verification.
        """)
        
        try:
            response = await self.llms['A2'].ainvoke(prompt.format(claim=claim))
            content = response.content
            
            entities = []
            entity_section = re.search(r'ENTITIES:(.*?)(?=ACTOR_RISK_SCORE|$)', content, re.DOTALL)
            if entity_section:
                entity_lines = entity_section.group(1).strip().split('\n')
                for line in entity_lines:
                    line = line.strip()
                    if line.startswith('-'):
                        entity_text = line[1:].strip()
                        if ':' in entity_text:
                            entity_name = entity_text.split(':')[0].strip()
                        else:
                            entity_name = entity_text.split('|')[0].strip() if '|' in entity_text else entity_text
                        if entity_name:
                            entities.append(entity_name)
                    elif line and not line.startswith('ACTOR_RISK_SCORE'):
                        entities.append(line)
            
            risk_match = re.search(r'ACTOR_RISK_SCORE:\s*([\d.]+)', content)
            actor_risk = float(risk_match.group(1)) if risk_match else 0.5
            
            lambda_weight = 0.5
            actor_type_prior = 0.7 if any('person' in ent.lower() for ent in entities) else 0.4
            actor_specific_prior = actor_risk
            s2 = lambda_weight * actor_type_prior + (1 - lambda_weight) * actor_specific_prior
            
            logger.info(f"A2: Found {len(entities)} entities with s2={s2:.3f}")
            
            return AgentResult(
                agent_id="A2", score=s2, 
                evidence=[f"Identified {len(entities)} key entities with risk assessment"],
                sources=[], 
                metadata={
                    "entities": entities, 
                    "actor_risk": actor_risk, 
                    "lambda_weight": lambda_weight
                }
            )
        except Exception as e:
            logger.error(f"A2 error: {e}")
            return AgentResult(
                "A2", 0.5, 
                [f"Error in entity resolution: {str(e)}"], 
                [], 
                {"entities": []}
            )

    # ============================================================================
    # AGENT A3: OFFICIAL-SOURCE VERIFIER (FIXED)
    # ============================================================================

    async def agent_a3_official_source_verifier(self, claim: str, entities: List[str], domain: str = "general") -> AgentResult:
        """A3: Official-Source Verifier - Uses Groq LLM to craft search queries and analyze evidence"""
        try:
            logger.info(f"A3: Starting official source verification for domain '{domain}'")
            
            # Step 1: Use A3 Groq LLM to generate precise search queries
            query_prompt = PromptTemplate.from_template("""
You are an expert fact-checker creating precise search queries for official source verification.

Generate 4 search queries for this claim: "{claim}"

**INTELLIGENT SITE SELECTION**: Choose appropriate sites based on claim category:

**POLITICS/GOVERNMENT**: site:pib.gov.in, site:pmindia.gov.in, site:india.gov.in, site:mygov.in
**BUSINESS/FINANCE**: site:reuters.com, site:bloomberg.com, site:forbes.com, site:moneycontrol.com, site:livemint.com
**HEALTH/SCIENCE**: site:who.int, site:mohfw.gov.in, site:icmr.gov.in, site:nature.com
**SPORTS**: site:bcci.tv, site:aiff.com, site:espn.in, site:sportskeeda.com
**ENTERTAINMENT**: site:filmfare.com, site:bollywoodhungama.com, site:pinkvilla.com
**GENERAL NEWS**: site:timesofindia.com, site:hindustantimes.com, site:indianexpress.com
**INTERNATIONAL**: site:bbc.com, site:cnn.com, site:aljazeera.com

Requirements:
1. **SMART SITE TARGETING**: Based on claim domain '{domain}', select 2-3 most relevant site: operators
2. Include fact-check variants with terms like "fact check", "verification", "debunked"  
3. Add 2025 filters using after:2025-01-01 when relevant
4. **CRITICAL**: Include real-time keywords: "real-time OR live OR latest OR current OR updated" for recent claims
5. For financial data, add terms like "net worth real-time", "latest valuation", "current market cap"

Format as:
QUERY1: [domain-specific official site query with real-time terms]
QUERY2: [fact check query with appropriate fact-check sites] 
QUERY3: [recent news with 2025 filter + live/latest keywords + domain-relevant sites]
QUERY4: [alternative verification query with current data terms + backup sites]

Claim domain: {domain}
            """)
            
            response = await self.llms['A3'].ainvoke(query_prompt.format(claim=claim, domain=domain))
            content = response.content
            logger.info(f"A3: Generated search queries using Groq LLM")
            
            # Parse generated queries
            queries = []
            for i in range(1, 5):
                query_match = re.search(rf'QUERY{i}:\s*(.+)', content, re.IGNORECASE)
                if query_match:
                    queries.append(query_match.group(1).strip())
            
            # Enhanced fallback queries with open sources
            if len(queries) < 4:
                current_year = datetime.now().year
                real_time_terms = "(real-time OR live OR latest OR current OR updated)"
                queries = [
                    f"site:wikipedia.org OR site:britannica.com OR site:pib.gov.in \"{claim}\" {real_time_terms}",
                    f"\"{claim}\" fact check debunk hoax site:factcheck.org OR site:snopes.com OR site:politifact.com",
                    f"\"{claim}\" after:2025-01-01 (latest OR current) site:reuters.com OR site:bbc.com OR site:apnews.com",
                    f"site:timesofindia.com OR site:thehindu.com OR site:ndtv.com OR site:firstpost.com \"{claim}\" {real_time_terms}"
                ]
            
            # Step 2: Perform searches
            all_sources = []
            search_results = []
            
            for query in queries[:4]:  # Limit to 4 queries
                try:
                    results = await self.web_search(query, max_results=5)
                    search_results.extend(results)
                    all_sources.extend(results)
                except Exception as e:
                    logger.warning(f"A3 search error for '{query[:50]}...': {e}")
            
            # Step 3: Use A3 Groq LLM to analyze top snippets
            if search_results:
                top_snippets = []
                for result in search_results[:10]:  # Analyze top 10 results
                    snippet_text = f"URL: {result.get('url', '')}\nTitle: {result.get('title', '')}\nSnippet: {result.get('snippet', '')}"
                    top_snippets.append(snippet_text)
                
                analysis_prompt = PromptTemplate.from_template("""
You are analyzing search results for fact-checking this claim: "{claim}"

Count supporting evidence (confirms the event/fact) and contradicting evidence (denies, debunks, or shows no record).

Search Results:
{snippets}

Analyze each result and count:
- SUPPORTING: Results that confirm or verify the claim
- CONTRADICTING: Results that deny, debunk, contradict, or show "no record" of the event
- NEUTRAL/TRENDING: For financial claims, older data showing positive trends (e.g., "$92B in April" + "growth" signals) should be counted as NEUTRAL, not contradicting

**Special Instructions for Business/Finance Claims:**
- If claim mentions current/recent values (2024-2025) but sources show older positive data with growth indicators
- Score older trending data as NEUTRAL rather than contradicting
- Example: Claim "$100B in Oct 2025" + Source "$92B in April + strong growth" = NEUTRAL trend support

For fabricated events (like fake inaugurations), absence of credible news coverage counts as contradiction.

Format:
SUPPORT_COUNT: [number]
CONTRADICT_COUNT: [number]
REASONING: [brief explanation of the counts]
                """)
                
                snippets_text = "\n\n".join(top_snippets)
                analysis_response = await self.llms['A3'].ainvoke(
                    analysis_prompt.format(claim=claim, snippets=snippets_text)
                )
                analysis_content = analysis_response.content
                logger.info(f"A3: Analyzed {len(top_snippets)} snippets using Groq LLM")
                
                # Parse analysis results
                support_match = re.search(r'SUPPORT_COUNT:\s*(\d+)', analysis_content)
                contradict_match = re.search(r'CONTRADICT_COUNT:\s*(\d+)', analysis_content)
                reasoning_match = re.search(r'REASONING:\s*(.+)', analysis_content, re.IGNORECASE)
                
                num_support = int(support_match.group(1)) if support_match else 0
                num_contradict = int(contradict_match.group(1)) if contradict_match else 0
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
                
            else:
                num_support = 0
                num_contradict = 0
                reasoning = "No search results found"
            
            # Step 4: Calculate s3 score - improved to reduce false negatives
            # Original: s3 = (num_contradict + 1) / (num_contradict + 1 + num_support + 2)
            if num_support > num_contradict:
                # More support than contradictions → favor Real (lower fake score)
                s3 = max(0.15, (num_contradict + 0.5) / (num_contradict + num_support + 4))
            elif num_contradict == 0 and num_support == 0:
                # No evidence either way → neutral (was too harsh before)
                s3 = 0.5
            else:
                # More contradictions → use conservative formula, cap at 0.75
                s3 = min(0.75, (num_contradict + 1) / (num_contradict + 1 + num_support + 2))
            
            # Additional cap if support > 1.5 * contradict (strong support)
            if num_support > 1.5 * num_contradict:
                s3 = min(s3, 0.2)
            
            logger.info(f"A3: Support={num_support}, Contradict={num_contradict}, s3={s3:.3f}")
            
            evidence = [
                f"Support evidence: {num_support} sources",
                f"Contradict evidence: {num_contradict} sources", 
                f"Analysis: {reasoning}"
            ]
            
            return AgentResult(
                agent_id="A3", 
                score=s3, 
                evidence=evidence,
                sources=all_sources[:10], 
                metadata={
                    "num_support": num_support, 
                    "num_contradict": num_contradict, 
                    "total_sources": len(all_sources),
                    "reasoning": reasoning
                }
            )
            
        except Exception as e:
            logger.error(f"A3 error: {e}")
            return AgentResult("A3", 0.6, [f"Error in official source verification: {str(e)}"], [], {})

    # ============================================================================
    # AGENT A4: PROPAGANDA AND TOXICITY DETECTOR
    # ============================================================================

    async def agent_a4_propaganda_toxicity_detector(self, claim: str) -> AgentResult:
        """A4: Enhanced Propaganda and Toxicity Detector with bias analysis"""
        try:
            logger.info(f"A4: Starting propaganda and toxicity analysis")
            
            prompt = PromptTemplate.from_template("""
You are an expert in detecting propaganda, bias, and toxic language in news claims.

Analyze this news claim for manipulation patterns and bias:

Claim: "{claim}"

Evaluate these aspects:
1. Emotional manipulation (inflammatory language, hyperbole)
2. Political bias (loaded terms, partisan framing)
3. Sensationalism (exaggerated claims, clickbait elements)
4. Hate speech indicators (divisive language, targeting groups)
5. Misinformation patterns (typical fake news structures)

Provide detailed analysis and scores:
EMOTIONAL_MANIPULATION: [0.0-1.0]
POLITICAL_BIAS: [0.0-1.0]
SENSATIONALISM: [0.0-1.0]
HATE_INDICATORS: [0.0-1.0]
MISINFORMATION_PATTERN: [0.0-1.0]

Overall assessment:
TOXICITY_SCORE: [0.0-1.0] (higher = more toxic/biased)

Explain your reasoning, especially for scores >0.3.
            """)
            
            response = await self.llms['A4'].ainvoke(prompt.format(claim=claim))
            content = response.content
            logger.info(f"A4: Completed bias analysis using Groq LLM")
            
            # Parse scores
            aspects = ['EMOTIONAL_MANIPULATION', 'POLITICAL_BIAS', 'SENSATIONALISM', 'HATE_INDICATORS', 'MISINFORMATION_PATTERN']
            aspect_scores = {}
            
            for aspect in aspects:
                match = re.search(rf'{aspect}:\s*([\d.]+)', content)
                aspect_scores[aspect] = float(match.group(1)) if match else 0.0
            
            toxicity_match = re.search(r'TOXICITY_SCORE:\s*([\d.]+)', content)
            toxicity_score = float(toxicity_match.group(1)) if toxicity_match else 0.5
            
            # Calculate s4 based on toxicity score
            if toxicity_score > 0.7:
                s4 = 0.8  # High toxicity indicates fake
            elif toxicity_score > 0.4:
                s4 = 0.6  # Medium toxicity
            else:
                s4 = toxicity_score  # Use the score directly for low toxicity
            
            logger.info(f"A4: Toxicity score={toxicity_score:.3f}, s4={s4:.3f}")
            
            evidence = [f"Toxicity analysis: {toxicity_score:.3f} overall score"]
            if toxicity_score > 0.3:
                high_aspects = [k for k, v in aspect_scores.items() if v > 0.3]
                if high_aspects:
                    evidence.append(f"High scores in: {', '.join(high_aspects)}")
            
            return AgentResult(
                agent_id="A4", 
                score=s4,
                evidence=evidence,
                sources=[], 
                metadata={
                    "aspect_scores": aspect_scores, 
                    "toxicity_score": toxicity_score,
                    "analysis": content
                }
            )
            
        except Exception as e:
            logger.error(f"A4 error: {e}")
            return AgentResult("A4", 0.5, [f"Error in propaganda detection: {str(e)}"], [], {})

    # ============================================================================
    # AGENT A5: RAG INCONSISTENCY AGENT (ENHANCED)
    # ============================================================================

    async def agent_a5_rag_inconsistency_agent(self, claim: str, entities: List[str], domain: str = "general") -> AgentResult:
        """A5: RAG Inconsistency Agent - Uses Groq LLM for historical query generation and analysis"""
        try:
            logger.info(f"A5: Starting historical context analysis for domain '{domain}'")
            current_year = datetime.now().year
            
            # Step 1: Use A5 Groq LLM to generate historical queries
            query_prompt = PromptTemplate.from_template("""
You are an expert researcher creating queries to find historical context and verification information.

Generate 4 historical verification queries for this claim: "{claim}"

Requirements:
1. Include 'verified', 'debunked', 'fact check' terms
2. Add 2025 filters and recent news searches  
3. Use site: operators for news archives and fact-check sites
4. For business claims, include financial news sites
5. **CRITICAL**: Include real-time search terms: "real-time OR live OR latest OR current OR updated"
6. For financial data, add "latest report", "current valuation", "real-time data"

Format as:
QUERY1: [historical verification query with real-time terms]
QUERY2: [debunk/fact-check query]
QUERY3: [recent news with 2025 filter + latest/current keywords]
QUERY4: [archive/verification query with current data terms]

Claim domain: {domain}
            """)
            
            response = await self.llms['A5'].ainvoke(query_prompt.format(claim=claim, domain=domain))
            content = response.content
            logger.info(f"A5: Generated historical queries using Groq LLM")
            
            # Parse generated queries
            queries = []
            for i in range(1, 5):
                query_match = re.search(rf'QUERY{i}:\s*(.+)', content, re.IGNORECASE)
                if query_match:
                    queries.append(query_match.group(1).strip())
            
            # Smart fallback: let A5 LLM generate domain-specific queries
            if len(queries) < 4:
                fallback_prompt = PromptTemplate.from_template("""
Create 4 historical verification queries for: "{claim}" (Domain: {domain})

**DOMAIN-SPECIFIC APPROACH**:
- **POLITICS**: Use government archives, political fact-checkers, parliamentary records
- **BUSINESS**: Use financial databases, company filings, business news archives  
- **HEALTH**: Use medical journals, WHO archives, health ministry records
- **SPORTS**: Use sports federations, tournament records, official league sites
- **ENTERTAINMENT**: Use industry databases, award records, box office data
- **GENERAL**: Use news archives, Wikipedia, general fact-checkers

Format:
QUERY1: [domain-specific historical verification]
QUERY2: [fact-check database search]  
QUERY3: [recent archives with real-time terms]
QUERY4: [alternative verification approach]

Include real-time terms: "real-time OR live OR latest OR current OR updated"
                """)
                
                try:
                    fallback_response = await self.llms['A5'].ainvoke(
                        fallback_prompt.format(claim=claim, domain=domain)
                    )
                    # Parse fallback queries
                    fallback_queries = []
                    for i in range(1, 5):
                        query_match = re.search(rf'QUERY{i}:\s*(.+)', fallback_response.content, re.IGNORECASE)
                        if query_match:
                            fallback_queries.append(query_match.group(1).strip())
                    
                    queries = fallback_queries if len(fallback_queries) >= 4 else [
                        f"\"{claim}\" verified OR debunked OR \"fact check\" (real-time OR latest)",
                        f"\"{claim}\" after:2025-01-01 site:wikipedia.org OR site:reuters.com OR site:bbc.com", 
                        f"\"{claim}\" hoax OR fake OR misinformation site:factcheck.org OR site:snopes.com",
                        f"site:timesofindia.com OR site:thehindu.com OR site:ndtv.com \"{claim}\" (latest OR current)"
                    ]
                except Exception as e:
                    logger.warning(f"A5 fallback LLM failed: {e}")
                    real_time_terms = "(real-time OR live OR latest OR current OR updated)"
                    queries = [
                        f"\"{claim}\" verified OR debunked OR \"fact check\" {real_time_terms}",
                        f"\"{claim}\" after:2025-01-01 site:wikipedia.org OR site:reuters.com OR site:bbc.com",
                        f"\"{claim}\" hoax OR fake OR misinformation site:factcheck.org OR site:snopes.com",
                        f"site:timesofindia.com OR site:thehindu.com OR site:ndtv.com \"{claim}\" {real_time_terms}"
                    ]
            
            # Step 2: Perform searches and collect passages
            all_sources = []
            all_passages = []
            
            for query in queries[:4]:  # Limit to 4 queries
                try:
                    results = await self.web_search(query, max_results=5, timelimit="y")
                    all_sources.extend(results)
                    
                    for result in results:
                        passage = f"URL: {result.get('url', '')}\nTitle: {result.get('title', '')}\nSnippet: {result.get('snippet', '')}"
                        all_passages.append(passage)
                        
                except Exception as e:
                    logger.warning(f"A5 search error for '{query[:50]}...': {e}")
            
            # Step 3: Use A5 Groq LLM to analyze passages
            if all_passages:
                analysis_prompt = PromptTemplate.from_template("""
You are analyzing search results for historical context verification of this claim: "{claim}"

Count supporting evidence (confirms/verifies the claim) and contradicting evidence (debunks/denies the claim).

Search Results:
{passages}

For fabricated or fake events, lack of credible historical coverage counts as contradiction.
For business/finance claims in 2025, consider recent sources more heavily.

**Special Instructions for Business/Finance Claims:**
- If claim mentions current/recent values (2024-2025) but sources show older positive data with growth trends
- DO NOT count older trending data as contradicting - count as NEUTRAL or soft support
- Example: Claim "$100B in Oct 2025" + Historical "$92B in April + consistent growth" = Trending Support
- Only count direct denials or debunks as true contradictions

Analyze and count:
- SUPPORT: Results that confirm, verify, or validate the claim (including positive trending data)
- CONTRADICT: Results that debunk, deny, contradict, or show no historical record (exclude older trending data)

Format:
SUPPORT: [number]
CONTRADICT: [number]
REASONING: [brief explanation]
                """)
                
                passages_text = "\n\n".join(all_passages[:10])  # Limit to 10 passages
                analysis_response = await self.llms['A5'].ainvoke(
                    analysis_prompt.format(claim=claim, passages=passages_text)
                )
                analysis_content = analysis_response.content
                logger.info(f"A5: Analyzed {len(all_passages)} passages using Groq LLM")
                
                # Parse analysis results
                support_match = re.search(r'SUPPORT:\s*(\d+)', analysis_content)
                contradict_match = re.search(r'CONTRADICT:\s*(\d+)', analysis_content)
                reasoning_match = re.search(r'REASONING:\s*(.+)', analysis_content, re.IGNORECASE)
                
                support_count = int(support_match.group(1)) if support_match else 0
                contradict_count = int(contradict_match.group(1)) if contradict_match else 0
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
                
                # 2x support boost for business/finance domain with 2025 content
                if domain == "business/finance":
                    passages_with_2025 = sum(1 for passage in all_passages if "2025" in passage)
                    if passages_with_2025 > 0:
                        support_count = int(support_count * 2)  # 2x boost
                        logger.info(f"A5: Applied 2x support boost for business/finance domain")
                
            else:
                support_count = 0
                contradict_count = 0
                reasoning = "No search results found"
            
            # Step 4: Calculate s5 score - improved historical context scoring
            # Original: s5 = (contradict_count + 1) / (contradict_count + 1 + support_count + 2)
            if support_count > contradict_count:
                # More historical support → favor Real (lower fake score)
                s5 = max(0.2, (contradict_count + 0.5) / (contradict_count + support_count + 4))
            elif contradict_count == 0 and support_count == 0:
                # No historical context → neutral but slightly favor Real for tech claims
                s5 = 0.4  # Reduced from 0.333
            else:
                # More contradictions → conservative scoring, cap at 0.8
                s5 = min(0.8, (contradict_count + 1) / (contradict_count + 1 + support_count + 2))
            
            logger.info(f"A5: Support={support_count}, Contradict={contradict_count}, s5={s5:.3f}")
            
            evidence = [
                f"Historical analysis: {support_count} supporting, {contradict_count} contradicting",
                f"Reasoning: {reasoning}"
            ]
            
            return AgentResult(
                agent_id="A5", 
                score=s5,
                evidence=evidence,
                sources=all_sources[:10], 
                metadata={
                    "support_count": support_count, 
                    "contradict_count": contradict_count, 
                    "total_sources": len(all_sources),
                    "reasoning": reasoning
                }
            )
            
        except Exception as e:
            logger.error(f"A5 error: {e}")
            return AgentResult("A5", 0.6, [f"Error in historical analysis: {str(e)}"], [], {})

    # ============================================================================
    # AGENT A6: ORCHESTRATOR AGENT (ENHANCED)
    # ============================================================================

    async def agent_a6_orchestrator(self, results: Dict[str, AgentResult], claim: str) -> FinalVerdict:
        """A6: Orchestrator Agent - Fuses all agent outputs with Groq LLM explanation"""
        try:
            logger.info(f"A6: Starting orchestration and fusion")
            
            scores = [results[f'A{i}'].score for i in range(1, 6)]
            domain = results['A1'].metadata.get('domain', 'general')
            entities = results['A2'].metadata.get('entities', [])
            
            # Enhanced base weights: [A1, A2, A3, A4, A5] = [0.5, 0.5, 1.5, 0.5, 2.0]
            base_weights = [0.5, 0.5, 1.5, 0.5, 2.0]
            
            # Business/finance domain adjustments: A3=1.8 (slight boost only)
            if domain == "business/finance":
                base_weights[2] = 1.8  # A3 slight boost for business
            
            # Calculate weighted fusion
            weighted_sum = sum(w * s for w, s in zip(base_weights, scores))
            weight_total = sum(base_weights)
            weighted_avg = weighted_sum / weight_total
            
            # Softer fusion with increased temperature
            temperature = 3.0
            L = (weighted_avg - 0.5) / temperature
            P_fake = 1 / (1 + math.exp(-L))
            
            # Historical override: if s5 < 0.2, subtract 0.1 from P_fake
            if scores[4] < 0.2:
                P_fake = max(0.0, P_fake - 0.1)
            
            # Enhanced confidence calculation
            score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0
            confidence_base = (1 - abs(0.5 - P_fake)) * 200
            confidence_penalty = (1 - min(score_variance, 0.5))
            confidence = int(confidence_base * confidence_penalty)
            confidence = max(70, min(95, confidence))  # Capped 70-95%
            
            # Adjust threshold based on domain and claim content
            if domain == "business/finance":
                threshold = 0.55  # Higher threshold for business claims
            elif any(keyword in claim.lower() for keyword in ['ai', 'gpt', 'openai', 'technology', 'software', 'release']):
                threshold = 0.52  # Slightly higher threshold for tech claims prone to false negatives
            else:
                threshold = 0.5   # Standard threshold
            
            final_verdict = "Fake" if P_fake > threshold else "Real"
            
            # Use A6 Groq LLM to generate explanation
            explanation_prompt = PromptTemplate.from_template("""
You are explaining the fusion verdict for a fact-checking system.

Claim: "{claim}"
Agent Scores: A1={s1:.3f}, A2={s2:.3f}, A3={s3:.3f}, A4={s4:.3f}, A5={s5:.3f}
Domain: {domain}
P_fake: {p_fake:.3f}
Confidence: {confidence}%
Final Verdict: {verdict}

Write a 1-sentence explanation of why this verdict was reached based on the scores and domain.
            """)
            
            explanation_response = await self.llms['A6'].ainvoke(
                explanation_prompt.format(
                    claim=claim,
                    s1=scores[0], s2=scores[1], s3=scores[2], s4=scores[3], s5=scores[4],
                    domain=domain, p_fake=P_fake, confidence=confidence, verdict=final_verdict
                )
            )
            explanation = explanation_response.content.strip()
            logger.info(f"A6: Generated explanation using Groq LLM")
            
            # Extract clean named entities
            named_entities = []
            if entities:
                for entity in entities:
                    entity_match = re.search(r'^([^:|]+)', entity.strip())
                    if entity_match:
                        clean_entity = entity_match.group(1).strip()
                        if clean_entity and clean_entity not in named_entities:
                            named_entities.append(clean_entity)
            
            # Compile fact findings from all agents + explanation
            fact_findings = [explanation]  # Add explanation first
            for i, result in enumerate([results[f'A{j}'] for j in range(1, 6)], 1):
                if result.evidence:
                    fact_findings.extend([f"A{i}: {evidence}" for evidence in result.evidence[:1]])
            
            # Collect unique sources
            all_sources = []
            for result in results.values():
                all_sources.extend(result.sources)
            
            unique_sources = []
            seen_urls = set()
            for source in all_sources:
                url = source.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(source)
            
            logger.info(f"A6: Final fusion P_fake={P_fake:.3f} → {final_verdict} ({confidence}%)")
            logger.info(f"A6: Score breakdown - A1:{scores[0]:.3f}, A2:{scores[1]:.3f}, A3:{scores[2]:.3f}, A4:{scores[3]:.3f}, A5:{scores[4]:.3f}")
            
            return FinalVerdict(
                named_entities=named_entities[:5], 
                domain=domain, 
                fact_finding=fact_findings,
                sources=unique_sources[:10], 
                final_verdict=final_verdict, 
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"A6 error: {e}")
            return FinalVerdict(
                named_entities=[], 
                domain="general", 
                fact_finding=[f"Error in orchestration: {str(e)}"], 
                sources=[], 
                final_verdict="Uncertain", 
                confidence=50
            )

    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================

    async def process_claim(self, claim: str) -> FinalVerdict:
        """Main processing pipeline - orchestrates all 6 agents with robust error handling"""
        try:
            logger.info(f"🚀 Processing claim: '{claim[:50]}...'")
            
            # Step 1: A2 first to get entities (with error handling)
            try:
                a2_result = await self.agent_a2_prime_actor_resolver(claim)
                entities = a2_result.metadata.get('entities', [])
            except Exception as e:
                logger.error(f"A2 failed: {e}")
                a2_result = AgentResult("A2", 0.6, [f"A2 error: {str(e)}"], [], {"entities": []})
                entities = []
            
            # Step 2: A1 and A4 in parallel (with error handling)
            a1_task = self.agent_a1_domain_router(claim)
            a4_task = self.agent_a4_propaganda_toxicity_detector(claim)
            
            try:
                a1_result, a4_result = await asyncio.gather(a1_task, a4_task, return_exceptions=True)
                
                if isinstance(a1_result, Exception):
                    logger.error(f"A1 failed: {a1_result}")
                    a1_result = AgentResult("A1", 0.6, [f"A1 error: {str(a1_result)}"], [], {"domain": "general"})
                    
                if isinstance(a4_result, Exception):
                    logger.error(f"A4 failed: {a4_result}")
                    a4_result = AgentResult("A4", 0.6, [f"A4 error: {str(a4_result)}"], [], {})
                    
            except Exception as e:
                logger.error(f"A1/A4 parallel execution failed: {e}")
                a1_result = AgentResult("A1", 0.6, [f"A1 error: {str(e)}"], [], {"domain": "general"})
                a4_result = AgentResult("A4", 0.6, [f"A4 error: {str(e)}"], [], {})
            
            domain = a1_result.metadata.get('domain', 'general')
            
            # Step 3: A3 and A5 in parallel with entities and domain context (with error handling)
            a3_task = self.agent_a3_official_source_verifier(claim, entities, domain)
            a5_task = self.agent_a5_rag_inconsistency_agent(claim, entities, domain)
            
            try:
                a3_result, a5_result = await asyncio.gather(a3_task, a5_task, return_exceptions=True)
                
                if isinstance(a3_result, Exception):
                    logger.error(f"A3 failed: {a3_result}")
                    a3_result = AgentResult("A3", 0.6, [f"A3 error: {str(a3_result)}"], [], {})
                    
                if isinstance(a5_result, Exception):
                    logger.error(f"A5 failed: {a5_result}")
                    a5_result = AgentResult("A5", 0.6, [f"A5 error: {str(a5_result)}"], [], {})
                    
            except Exception as e:
                logger.error(f"A3/A5 parallel execution failed: {e}")
                a3_result = AgentResult("A3", 0.6, [f"A3 error: {str(e)}"], [], {})
                a5_result = AgentResult("A5", 0.6, [f"A5 error: {str(e)}"], [], {})
            
            # Step 4: A6 orchestration (with error handling)
            results = {
                'A1': a1_result, 
                'A2': a2_result, 
                'A3': a3_result, 
                'A4': a4_result, 
                'A5': a5_result
            }
            
            try:
                final_verdict = await self.agent_a6_orchestrator(results, claim)
            except Exception as e:
                logger.error(f"A6 orchestration failed: {e}")
                final_verdict = FinalVerdict(
                    named_entities=[], 
                    domain=domain, 
                    fact_finding=[f"Pipeline error: {str(e)}"], 
                    sources=[], 
                    final_verdict="Uncertain", 
                    confidence=50
                )
            
            logger.info(f"✅ Pipeline completed: {final_verdict.final_verdict} ({final_verdict.confidence}%)")
            return final_verdict
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return FinalVerdict(
                named_entities=[], 
                domain="general", 
                fact_finding=[f"Pipeline error: {str(e)}"], 
                sources=[], 
                final_verdict="Uncertain", 
                confidence=50
            )

   