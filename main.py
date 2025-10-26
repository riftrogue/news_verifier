#!/usr/bin/env python3
"""
Main Processing Pipeline for Multi-Agent Fact-Checking System

This is the complete fact-checking system with both JSON API and pretty console output.
Unified interface for all fact-checking operations.

Usage examples:
  - Pretty console output:
      python main.py "your claim to verify"
  - JSON output (machine-readable):
      python main.py --json "your claim to verify"
  - Save JSON output to verified reports:
      python main.py --json --save "your claim to verify"
"""

import os
import sys
import json
import argparse
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_header():
    """Print a nice header for the tool"""
    print("=" * 80)
    print("🔍 MULTI-AGENT FACT-CHECKING ANALYSIS")
    print("=" * 80)

def print_separator():
    """Print a section separator"""
    print("-" * 80)

def format_confidence_bar(confidence):
    """Create a visual confidence bar"""
    bar_length = 20
    filled = int((confidence / 100) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    return f"[{bar}] {confidence}%"

def get_verdict_emoji(verdict):
    """Get emoji based on verdict"""
    if verdict.lower() == "real":
        return "✅"
    elif verdict.lower() == "fake":
        return "❌"
    else:
        return "⚠️"

def check_env() -> Dict[str, bool]:
    """Check for required API keys and return availability map"""
    keys = {
        "GROQ_API_KEY_A1": bool(os.getenv("GROQ_API_KEY_A1")),
        "GROQ_API_KEY_A2": bool(os.getenv("GROQ_API_KEY_A2")),
        "GROQ_API_KEY_A3": bool(os.getenv("GROQ_API_KEY_A3")),
        "GROQ_API_KEY_A6": bool(os.getenv("GROQ_API_KEY_A6")),
        "GEMINI_API_KEY_A4": bool(os.getenv("GEMINI_API_KEY_A4")),
        "GEMINI_API_KEY_A5": bool(os.getenv("GEMINI_API_KEY_A5")),
    }
    return keys

class FactCheckingPipeline:
    """Main processing pipeline for the multi-agent fact-checking system"""
    
    def __init__(self):
        """Initialize the fact-checking pipeline"""
        from llm_agents import MultiAgentFactChecker
        self.fact_checker = MultiAgentFactChecker()
    
    async def process_claim(self, news: str) -> Dict[str, Any]:
        """Main processing pipeline - processes a news claim through all 6 agents"""
        try:
            result = await self.fact_checker.process_claim(news)
            formatted_result = self._format_results(news, result)
            return formatted_result
        except Exception as e:
            return self._error_response(news, str(e))
    
    def _format_results(self, news: str, result) -> Dict[str, Any]:
        """Format multi-agent results into standardized output structure"""
        return {
            "news": news, "verdict": result.final_verdict.lower() == "real", "confidence": result.confidence / 100.0,
            "context": "multi_agent", "category": result.domain, "bias": "none",
            "verified_sources": [source.get("url", "") for source in result.sources],
            "entity_details": {"name": ", ".join(result.named_entities[:3]) if result.named_entities else "Unknown",
                              "label": None, "summary": None, "url": None, "source": "multi_agent_analysis"},
            "trace": [{"source": source.get("title", "Unknown Source"), "snippet": source.get("snippet", ""),
                      "url": source.get("url", "")} for source in result.sources[:5]],
            "explanation": (f"Multi-agent analysis: {result.final_verdict} with {result.confidence}% confidence. "
                           f"Found {len(result.sources)} sources. Key findings: {'; '.join(result.fact_finding[:2])}"),
            "multi_agent_details": {"domain": result.domain, "named_entities": result.named_entities,
                                  "fact_findings": result.fact_finding, "agent_count": 6, "analysis_type": "parallel_fusion"}
        }
    
    def _error_response(self, news: str, error_msg: str) -> Dict[str, Any]:
        """Generate standardized error response"""
        return {
            "news": news, "error": f"Multi-agent analysis failed: {error_msg}", "verdict": "Uncertain",
            "confidence": 0.5, "context": "error", "category": "general", "bias": "none", "verified_sources": [],
            "entity_details": {"name": "Unknown", "label": None, "summary": None, "url": None, "source": None},
            "trace": [], "explanation": f"Analysis failed due to: {error_msg}"
        }
    
    def save_to_verified_reports(self, result: Dict[str, Any], reports_path: str = "data/verified_reports.json") -> None:
        """Save analysis results to verified reports file"""
        try:
            os.makedirs(os.path.dirname(reports_path), exist_ok=True)
            try:
                with open(reports_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except FileNotFoundError:
                existing = []
            existing.append(result)
            with open(reports_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

pipeline = FactCheckingPipeline()

async def process_claim_with_multi_agent(news: str) -> Dict[str, Any]:
    """Main entry point for fact-checking claims"""
    return await pipeline.process_claim(news)

def save_to_verified_reports(result: Dict[str, Any], reports_path: str = "data/verified_reports.json") -> None:
    """Save results"""
    pipeline.save_to_verified_reports(result, reports_path)

async def analyze_claim_pretty(claim: str) -> bool:
    """Analyze a claim and print a human-friendly report to console"""
    env_ok = check_env()
    missing = [k for k, ok in env_ok.items() if not ok]
    if missing:
        print("⚠️  Missing environment variables detected:")
        for k in missing:
            print(f"   - {k}")
        print("   The analysis may be limited or fail for some agents.")
        print()
    
    print_header()
    print(f"📰 CLAIM: {claim}")
    print_separator()
    
    try:
        print("🚀 Initializing Multi-Agent System...")
        print("✅ All 6 agents ready! Processing claim...\n")

        from llm_agents import MultiAgentFactChecker
        fact_checker = MultiAgentFactChecker()
        result = await fact_checker.process_claim(claim)
        
        print("🎯 VERDICT & CONFIDENCE")
        verdict_emoji = get_verdict_emoji(result.final_verdict)
        print(f"   {verdict_emoji} FINAL VERDICT: {result.final_verdict.upper()}")
        print(f"   📊 CONFIDENCE: {format_confidence_bar(result.confidence)}")
        print(f"   📂 DOMAIN: {result.domain.title()}")
        print()
        
        if result.named_entities:
            print("👥 IDENTIFIED ENTITIES")
            for i, entity in enumerate(result.named_entities[:5], 1):
                print(f"   {i}. {entity}")
            print()
        
        if result.fact_finding:
            print("📋 AGENT ANALYSIS FINDINGS")
            for i, finding in enumerate(result.fact_finding, 1):
                print(f"   {i}. {finding}")
            print()
        
        if result.sources:
            print("🔗 VERIFIED SOURCES & REFERENCES")
            print(f"   Total Sources Found: {len(result.sources)}")
            print()
            for i, source in enumerate(result.sources[:10], 1):
                title = source.get('title', 'No Title Available')
                url = source.get('url', 'No URL')
                snippet = source.get('snippet', 'No description available')
                print(f"   [{i}] {title}")
                print(f"       🌐 URL: {url}")
                if snippet and len(snippet) > 5:
                    display_snippet = snippet[:150] + "..." if len(snippet) > 150 else snippet
                    print(f"       📄 Preview: {display_snippet}")
                print()
        else:
            print("🔗 SOURCES & REFERENCES")
            print("   ⚠️  No sources found for this claim")
            print()
        
        print("🔧 TECHNICAL ANALYSIS DETAILS")
        print(f"   • Agent System: 6-Agent Parallel Processing")
        print(f"   • Analysis Type: Mathematical Fusion")
        print(f"   • Domain Classification: {result.domain}")
        print(f"   • Named Entity Count: {len(result.named_entities)}")
        print(f"   • Source Verification Count: {len(result.sources)}")
        print(f"   • Processing Method: Async Multi-Agent")
        print()
        
        print("📝 ANALYSIS SUMMARY")
        if result.confidence >= 80:
            confidence_level, confidence_emoji = "High", "🔥"
        elif result.confidence >= 60:
            confidence_level, confidence_emoji = "Medium", "⚖️"
        else:
            confidence_level, confidence_emoji = "Low", "⚠️"
        print(f"   {confidence_emoji} Confidence Level: {confidence_level}")
        print(f"   📊 The multi-agent system analyzed this claim with {result.confidence}% confidence")
        print(f"   🎯 Final determination: {result.final_verdict.upper()}")
        if result.sources:
            print(f"   🔍 Analysis based on {len(result.sources)} verified sources")
        
        print_separator()
        print("✅ Analysis Complete!")
    except Exception as e:
        print("❌ ERROR DURING ANALYSIS")
        print(f"   Error Details: {str(e)}")
        print("   Please check your API keys and internet connection.")
        print_separator()
        return False
    return True

async def analyze_claim_json(claim: str, save: bool = False) -> Dict[str, Any]:
    """Analyze a claim and return machine-readable JSON; optionally save"""
    final_out = await process_claim_with_multi_agent(claim)
    if save:
        try:
            save_to_verified_reports(final_out)
        except Exception:
            pass
    return final_out

def cli_main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-Agent Fact-Checking System")
    parser.add_argument("claim", nargs="*", help="The claim to verify")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--save", action="store_true", help="Save JSON result to verified reports (implies --json)")
    args = parser.parse_args()

    if args.claim:
        claim = " ".join(args.claim)
    else:
        try:
            with open("data/temp_input.json", "r", encoding="utf-8") as f:
                tmp = json.load(f)
                claim = tmp.get("news") or ""
        except Exception:
            claim = ""

    if not claim:
        if args.json or args.save:
            print(json.dumps({"error": "No input provided. Pass as CLI args or set data/temp_input.json with key 'news'"}, ensure_ascii=False))
        else:
            print("❌ No claim provided. Usage:")
            print('   python main.py "your claim to verify"')
            print('   python main.py --json "your claim to verify"')
        return

    try:
        if args.json or args.save:
            result = asyncio.run(analyze_claim_json(claim, save=args.save))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            asyncio.run(analyze_claim_pretty(claim))
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    cli_main()