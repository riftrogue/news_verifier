import asyncio
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

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
    metadata: Dict

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
    A1: Domain Router
    A2: Prime-Actor Resolver 
    A3: Official-Source Verifier
    A4: Propaganda and Toxicity Detector
    A5: RAG Inconsistency Agent
    A6: Orchestrator Agent
    """
    
    def __init__(self):
        self.tavily = TavilySearch(max_results=5)
        
        # Initialize LLMs for each agent with specific API keys
        self.llms = {
            'A1': ChatGroq(
                api_key=os.getenv("GROQ_API_KEY_A1"),
                model="llama-3.1-8b-instant",
                temperature=0.1
            ),
            'A2': ChatGroq(
                api_key=os.getenv("GROQ_API_KEY_A2"),
                model="llama-3.1-8b-instant",
                temperature=0.1
            ),
            'A3': ChatGroq(
                api_key=os.getenv("GROQ_API_KEY_A3"),
                model="llama-3.1-70b-versatile",
                temperature=0.1
            ),
            'A4': ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GEMINI_API_KEY_A4"),
                model="gemini-2.5-flash",
                temperature=0.1
            ),
            'A5': ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GEMINI_API_KEY_A5"),
                model="gemini-2.5-flash",
                temperature=0.1
            ),
            'A6': ChatGroq(
                api_key=os.getenv("GROQ_API_KEY_A6"),
                model="llama-3.1-70b-versatile",
                temperature=0.1
            )
        }
        
        # Domain-specific fake news priors (from research)
        self.domain_priors = {
            'politics': 0.6,
            'sports': 0.4,
            'entertainment': 0.4,
            'science/health': 0.4,
            'general': 0.4
        }

    async def agent_a1_domain_router(self, claim: str) -> AgentResult:
        """
        A1: Domain Router - Classifies claim into domains and computes domain-based fake prior
        """
        prompt = PromptTemplate.from_template("""
You are a domain classification expert for Indian news claims. 

Classify this news claim into ONE of these domains: politics, sports, entertainment, science/health, general

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
- "Weather update for tomorrow" → general

Respond with ONLY the format above.
        """)
        
        try:
            response = await self.llms['A1'].ainvoke(prompt.format(claim=claim))
            content = response.content
            
            # Parse domain from response
            domain_match = re.search(r'DOMAIN:\s*(\w+(?:/\w+)?)', content)
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content)
            reasoning_match = re.search(r'REASONING:\s*(.+)', content)
            
            domain = domain_match.group(1) if domain_match else 'general'
            domain_confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
            
            # Compute Laplace smoothed prior: s_1 = P(Fake | domain)
            mu = 1  # Laplace smoothing parameter
            fake_rate = self.domain_priors.get(domain, 0.4)
            # Simulate training data counts (N_k=100, F_k based on fake_rate)
            N_k = 100
            F_k = int(fake_rate * N_k)
            s1 = (F_k + mu) / (N_k + 2 * mu)
            
            logger.info(f"A1: Classified '{claim[:50]}...' as {domain} with s1={s1:.3f}")
            
            return AgentResult(
                agent_id="A1",
                score=s1,
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
            logger.error(f"A1 error: {e}")
            return AgentResult("A1", 0.5, [f"Error in domain classification: {str(e)}"], [], {"domain": "general"})

    async def agent_a2_prime_actor_resolver(self, claim: str) -> AgentResult:
        """
        A2: Prime-Actor Resolver - Identifies key entities and resolves them
        """
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
            
            # Extract entities
            entities = []
            entity_section = re.search(r'ENTITIES:(.*?)(?=ACTOR_RISK_SCORE|$)', content, re.DOTALL)
            if entity_section:
                entity_lines = entity_section.group(1).strip().split('\n')
                for line in entity_lines:
                    if line.strip().startswith('-'):
                        entities.append(line.strip()[1:].strip())
            
            # Extract risk score
            risk_match = re.search(r'ACTOR_RISK_SCORE:\s*([\d.]+)', content)
            actor_risk = float(risk_match.group(1)) if risk_match else 0.5
            
            # Compute s2 using Bayesian smoothing
            lambda_weight = 0.5
            actor_type_prior = 0.7 if any('PERSON' in ent for ent in entities) else 0.4
            actor_specific_prior = actor_risk
            s2 = lambda_weight * actor_type_prior + (1 - lambda_weight) * actor_specific_prior
            
            logger.info(f"A2: Found {len(entities)} entities with s2={s2:.3f}")
            
            return AgentResult(
                agent_id="A2",
                score=s2,
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
            return AgentResult("A2", 0.5, [f"Error in entity resolution: {str(e)}"], [], {"entities": []})

    async def agent_a3_official_source_verifier(self, claim: str, entities: List[str]) -> AgentResult:
        """
        A3: Official-Source Verifier - Checks official sources using Tavily
        """
        try:
            # Search official Indian government and institutional sources
            official_queries = [
                f"site:pib.gov.in {claim}",
                f"site:pmindia.gov.in {claim}",
                f"site:india.gov.in {claim}",
                f"site:mygov.in {claim}"
            ]
            
            all_sources = []
            support_evidence = []
            contradict_evidence = []
            
            for query in official_queries[:2]:  # Limit to 2 searches to avoid rate limits
                try:
                    results = self.tavily.invoke({"query": query})
                    for result in results.get("results", []):
                        all_sources.append({
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "snippet": result.get("content", "")[:200] + "..."
                        })
                        
                        # Simple support/contradiction detection
                        snippet_lower = result.get("content", "").lower()
                        claim_words = set(claim.lower().split())
                        snippet_words = set(snippet_lower.split())
                        
                        overlap = len(claim_words.intersection(snippet_words)) / len(claim_words)
                        if overlap > 0.3:
                            support_evidence.append(result.get("title", ""))
                        elif any(neg in snippet_lower for neg in ["no", "not", "false", "denied", "refuted"]):
                            contradict_evidence.append(result.get("title", ""))
                            
                except Exception as search_error:
                    logger.warning(f"A3 search error for '{query}': {search_error}")
                    continue
            
            # Compute s3 = 1 - Σ w(e) [support V contradict]
            total_evidence = len(support_evidence) + len(contradict_evidence)
            if total_evidence == 0:
                s3 = 0.6  # No evidence found suggests higher fake probability
            else:
                support_weight = len(support_evidence) / total_evidence
                contradict_weight = len(contradict_evidence) / total_evidence
                s3 = 1 - (support_weight - contradict_weight)  # More support = lower fake probability
                s3 = max(0.0, min(1.0, s3))  # Clamp to [0,1]
            
            evidence_summary = []
            if support_evidence:
                evidence_summary.append(f"Found {len(support_evidence)} supporting official sources")
            if contradict_evidence:
                evidence_summary.append(f"Found {len(contradict_evidence)} contradicting official sources")
            if not support_evidence and not contradict_evidence:
                evidence_summary.append("No official sources found supporting or contradicting the claim")
            
            logger.info(f"A3: Found {len(all_sources)} official sources with s3={s3:.3f}")
            
            return AgentResult(
                agent_id="A3",
                score=s3,
                evidence=evidence_summary,
                sources=all_sources[:5],  # Limit to top 5
                metadata={
                    "support_count": len(support_evidence),
                    "contradict_count": len(contradict_evidence),
                    "total_sources": len(all_sources)
                }
            )
            
        except Exception as e:
            logger.error(f"A3 error: {e}")
            return AgentResult("A3", 0.6, [f"Error in official source verification: {str(e)}"], [], {})

    async def agent_a4_propaganda_toxicity_detector(self, claim: str) -> AgentResult:
        """
        A4: Propaganda and Toxicity Detector - Analyzes for manipulation patterns
        """
        prompt = PromptTemplate.from_template("""
You are an expert in detecting propaganda techniques and toxicity in news claims.

Analyze this claim for propaganda techniques and toxicity indicators:

Claim: "{claim}"

Check for these patterns:
1. Loaded/emotional language (words meant to provoke strong emotions)
2. Absolute statements without qualifiers ("all", "never", "always") 
3. Appeals to fear or anger
4. Unsubstantiated superlatives ("largest", "first ever", "unprecedented")
5. Divisive language targeting specific communities
6. Sensational or clickbait-style phrasing

Rate each category (0.0-1.0):
EMOTIONAL_LANGUAGE: [score]
ABSOLUTE_STATEMENTS: [score]
FEAR_APPEAL: [score]
UNSUBSTANTIATED_CLAIMS: [score]
DIVISIVE_LANGUAGE: [score]
SENSATIONAL_STYLE: [score]

OVERALL_TOXICITY_PROBABILITY: [0.0-1.0]
EXPLANATION: [brief reasoning]

Be objective and focus on linguistic patterns that correlate with misinformation.
        """)
        
        try:
            response = await self.llms['A4'].ainvoke(prompt.format(claim=claim))
            content = response.content
            
            # Extract scores
            scores = {}
            for category in ["EMOTIONAL_LANGUAGE", "ABSOLUTE_STATEMENTS", "FEAR_APPEAL", 
                           "UNSUBSTANTIATED_CLAIMS", "DIVISIVE_LANGUAGE", "SENSATIONAL_STYLE"]:
                match = re.search(f'{category}:\\s*([\\d.]+)', content)
                scores[category] = float(match.group(1)) if match else 0.0
            
            # Extract overall toxicity
            toxicity_match = re.search(r'OVERALL_TOXICITY_PROBABILITY:\s*([\d.]+)', content)
            P_hate = float(toxicity_match.group(1)) if toxicity_match else 0.0
            
            explanation_match = re.search(r'EXPLANATION:\s*(.+)', content)
            explanation = explanation_match.group(1) if explanation_match else "No explanation provided"
            
            # Compute s4 = β * P_hate
            beta = 1.2  # Hyperparameter for toxicity/fake correlation
            s4 = min(1.0, beta * P_hate)  # Clamp to max 1.0
            
            evidence = [f"Toxicity probability: {P_hate:.2f}", explanation]
            if max(scores.values()) > 0.5:
                high_scores = [k for k, v in scores.items() if v > 0.5]
                evidence.append(f"High scores in: {', '.join(high_scores)}")
            
            logger.info(f"A4: Detected toxicity P_hate={P_hate:.3f} with s4={s4:.3f}")
            
            return AgentResult(
                agent_id="A4",
                score=s4,
                evidence=evidence,
                sources=[],
                metadata={
                    "scores": scores,
                    "P_hate": P_hate,
                    "beta": beta,
                    "explanation": explanation
                }
            )
            
        except Exception as e:
            logger.error(f"A4 error: {e}")
            return AgentResult("A4", 0.3, [f"Error in toxicity detection: {str(e)}"], [], {})

    async def agent_a5_rag_inconsistency_agent(self, claim: str) -> AgentResult:
        """
        A5: RAG Inconsistency Agent - Checks against historical context using Tavily
        """
        try:
            # Search for historical context and recent news
            current_year = datetime.now().year
            historical_queries = [
                f'"{claim}" since:{current_year-1}-01-01',
                f'"{claim}" fact check',
                f'"{claim}" verification news'
            ]
            
            all_sources = []
            supporting_passages = []
            contradicting_passages = []
            
            for query in historical_queries:
                try:
                    results = self.tavily.invoke({"query": query})
                    for result in results.get("results", []):
                        all_sources.append({
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "snippet": result.get("content", "")[:200] + "..."
                        })
                        
                        # Analyze for support/contradiction
                        content_lower = result.get("content", "").lower()
                        title_lower = result.get("title", "").lower()
                        
                        # Look for fact-checking keywords
                        if any(word in content_lower or word in title_lower for word in 
                               ["fake", "false", "hoax", "misinformation", "debunked"]):
                            contradicting_passages.append(result.get("title", ""))
                        elif any(word in content_lower or word in title_lower for word in 
                                ["confirmed", "verified", "official", "announced"]):
                            supporting_passages.append(result.get("title", ""))
                            
                except Exception as search_error:
                    logger.warning(f"A5 search error for '{query}': {search_error}")
                    continue
            
            # Compute s5 = 1 - Σ w(e) [support V contradict]
            total_passages = len(supporting_passages) + len(contradicting_passages)
            if total_passages == 0:
                s5 = 0.5  # No historical evidence
            else:
                support_weight = len(supporting_passages) / total_passages
                contradict_weight = len(contradicting_passages) / total_passages
                s5 = contradict_weight  # More contradictions = higher fake probability
                s5 = max(0.0, min(1.0, s5))
            
            # Apply threshold τ = 0.5 for verdict
            tau = 0.5
            preliminary_verdict = "Fake" if s5 > tau else "Real"
            
            evidence = []
            if supporting_passages:
                evidence.append(f"Found {len(supporting_passages)} historical sources supporting the claim")
            if contradicting_passages:
                evidence.append(f"Found {len(contradicting_passages)} sources contradicting or fact-checking the claim")
            if not supporting_passages and not contradicting_passages:
                evidence.append("No clear historical evidence found for verification")
            
            logger.info(f"A5: Historical analysis yields s5={s5:.3f}, preliminary verdict: {preliminary_verdict}")
            
            return AgentResult(
                agent_id="A5",
                score=s5,
                evidence=evidence,
                sources=all_sources[:5],
                metadata={
                    "supporting_count": len(supporting_passages),
                    "contradicting_count": len(contradicting_passages),
                    "threshold": tau,
                    "preliminary_verdict": preliminary_verdict
                }
            )
            
        except Exception as e:
            logger.error(f"A5 error: {e}")
            return AgentResult("A5", 0.5, [f"Error in historical consistency check: {str(e)}"], [], {})

    async def agent_a6_orchestrator(self, claim: str, agent_results: List[AgentResult]) -> FinalVerdict:
        """
        A6: Orchestrator Agent - Fuses all agent outputs into final verdict
        """
        try:
            # Extract domain and entities from A1 and A2
            domain = "general"
            named_entities = []
            
            for result in agent_results:
                if result.agent_id == "A1":
                    domain = result.metadata.get("domain", "general")
                elif result.agent_id == "A2":
                    entities_raw = result.metadata.get("entities", [])
                    # Clean entity names
                    for ent in entities_raw:
                        if ':' in ent:
                            entity_name = ent.split(':')[0].strip()
                        else:
                            entity_name = ent.strip("- ")
                        if entity_name:
                            named_entities.append(entity_name)
            
            # Compute fusion: L = b + Σ w_i * s_i
            b = 0  # bias term
            weights = [1.0] * 5  # equal weights for A1-A5
            scores = [result.score for result in agent_results if result.agent_id != "A6"]
            
            # Ensure we have exactly 5 scores
            while len(scores) < 5:
                scores.append(0.5)  # default score for missing agents
            scores = scores[:5]  # truncate if more than 5
            
            L = b + sum(w * s for w, s in zip(weights, scores))
            
            # Apply sigmoid: P_fake = 1 / (1 + e^(-L))
            P_fake = 1 / (1 + math.exp(-L))
            
            # Final verdict with threshold 0.5
            threshold = 0.5
            final_verdict = "Fake" if P_fake > threshold else "Real"
            
            # Confidence calculation: (1 - |0.5 - P_fake|) * 200%
            confidence = int((1 - abs(0.5 - P_fake)) * 200)
            confidence = max(50, min(99, confidence))  # Clamp between 50-99%
            
            # Compile fact findings from all agents
            fact_finding = []
            all_sources = []
            
            for result in agent_results:
                fact_finding.extend(result.evidence)
                all_sources.extend(result.sources)
            
            # Remove duplicates and limit
            unique_sources = []
            seen_urls = set()
            for source in all_sources:
                url = source.get("url", "")
                if url and url not in seen_urls:
                    unique_sources.append(source)
                    seen_urls.add(url)
                if len(unique_sources) >= 5:
                    break
            
            # Limit fact findings to top 3
            fact_finding = fact_finding[:3] if len(fact_finding) > 3 else fact_finding
            
            logger.info(f"A6: Final fusion P_fake={P_fake:.3f} → {final_verdict} ({confidence}%)")
            
            return FinalVerdict(
                named_entities=named_entities,
                domain=domain,
                fact_finding=fact_finding,
                sources=unique_sources,
                final_verdict=final_verdict,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"A6 error: {e}")
            # Fallback verdict
            return FinalVerdict(
                named_entities=["Unknown"],
                domain="general",
                fact_finding=[f"Error in orchestration: {str(e)}"],
                sources=[],
                final_verdict="Uncertain",
                confidence=50
            )

    async def process_claim(self, claim: str) -> FinalVerdict:
        """
        Main processing pipeline: parallel A1-A5, then sequential A6
        """
        logger.info(f"Processing claim: '{claim[:100]}...'")
        
        # Phase 1: Run A1-A5 in parallel
        async def run_agent_a1():
            return await self.agent_a1_domain_router(claim)
        
        async def run_agent_a2():
            return await self.agent_a2_prime_actor_resolver(claim)
        
        async def run_agent_a3():
            # Need entities from A2, but we'll run in parallel and pass empty list for now
            return await self.agent_a3_official_source_verifier(claim, [])
        
        async def run_agent_a4():
            return await self.agent_a4_propaganda_toxicity_detector(claim)
        
        async def run_agent_a5():
            return await self.agent_a5_rag_inconsistency_agent(claim)
        
        # Execute A1-A5 in parallel
        parallel_tasks = [
            run_agent_a1(),
            run_agent_a2(), 
            run_agent_a3(),
            run_agent_a4(),
            run_agent_a5()
        ]
        
        agent_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
        
        # Handle any exceptions
        valid_results = []
        for i, result in enumerate(agent_results):
            if isinstance(result, Exception):
                logger.error(f"Agent A{i+1} failed: {result}")
                # Create dummy result
                valid_results.append(AgentResult(f"A{i+1}", 0.5, [f"Agent failed: {str(result)}"], [], {}))
            else:
                valid_results.append(result)
        
        # Phase 2: Run A6 sequentially with results from A1-A5
        final_verdict = await self.agent_a6_orchestrator(claim, valid_results)
        
        return final_verdict
# All legacy compatibility code removed as part of cleanup.

