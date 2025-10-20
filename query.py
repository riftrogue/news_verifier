#!/usr/bin/env python3
"""
Query Tool for Multi-Agent Fact-Checking System
Usage: python query.py "your claim to verify"
Provides detailed structured analysis with sources and references
"""

import sys
import asyncio
from llm_agents import MultiAgentFactChecker

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

async def analyze_claim(claim: str):
    """Analyze a single claim and display detailed results"""
    
    print_header()
    print(f"📰 CLAIM: {claim}")
    print_separator()
    
    try:
        # Initialize the fact checker
        print("🚀 Initializing Multi-Agent System...")
        fact_checker = MultiAgentFactChecker()
        print("✅ All 6 agents ready! Processing claim...\n")
        
        # Process the claim through all 6 agents
        result = await fact_checker.process_claim(claim)
        
        # Main Verdict Section
        print("🎯 VERDICT & CONFIDENCE")
        verdict_emoji = get_verdict_emoji(result.final_verdict)
        print(f"   {verdict_emoji} FINAL VERDICT: {result.final_verdict.upper()}")
        print(f"   📊 CONFIDENCE: {format_confidence_bar(result.confidence)}")
        print(f"   📂 DOMAIN: {result.domain.title()}")
        print()
        
        # Entities Section
        if result.named_entities:
            print("👥 IDENTIFIED ENTITIES")
            for i, entity in enumerate(result.named_entities[:5], 1):
                print(f"   {i}. {entity}")
            print()
        
        # Key Findings Section
        if result.fact_finding:
            print("📋 AGENT ANALYSIS FINDINGS")
            for i, finding in enumerate(result.fact_finding, 1):
                print(f"   {i}. {finding}")
            print()
        
        # Sources & References Section
        if result.sources:
            print("🔗 VERIFIED SOURCES & REFERENCES")
            print(f"   Total Sources Found: {len(result.sources)}")
            print()
            
            for i, source in enumerate(result.sources[:10], 1):  # Show up to 10 sources
                title = source.get('title', 'No Title Available')
                url = source.get('url', 'No URL')
                snippet = source.get('snippet', 'No description available')
                
                print(f"   [{i}] {title}")
                print(f"       🌐 URL: {url}")
                if snippet and len(snippet) > 5:
                    # Truncate snippet if too long
                    display_snippet = snippet[:150] + "..." if len(snippet) > 150 else snippet
                    print(f"       📄 Preview: {display_snippet}")
                print()
        else:
            print("🔗 SOURCES & REFERENCES")
            print("   ⚠️  No sources found for this claim")
            print()
        
        # Technical Details Section
        print("🔧 TECHNICAL ANALYSIS DETAILS")
        print(f"   • Agent System: 6-Agent Parallel Processing")
        print(f"   • Analysis Type: Mathematical Fusion")
        print(f"   • Domain Classification: {result.domain}")
        print(f"   • Named Entity Count: {len(result.named_entities)}")
        print(f"   • Source Verification Count: {len(result.sources)}")
        print(f"   • Processing Method: Async Multi-Agent")
        print()
        
        # Summary Section
        print("📝 ANALYSIS SUMMARY")
        if result.confidence >= 80:
            confidence_level = "High"
            confidence_emoji = "🔥"
        elif result.confidence >= 60:
            confidence_level = "Medium"
            confidence_emoji = "⚖️"
        else:
            confidence_level = "Low"
            confidence_emoji = "⚠️"
            
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

def main():
    """Main function to handle command line arguments"""
    
    # Check if claim is provided as argument
    if len(sys.argv) < 2:
        print("❌ Error: No claim provided!")
        print("\n📖 Usage:")
        print("   python query.py \"your claim to verify\"")
        print("\n💡 Examples:")
        print("   python query.py \"Elon Musk bought Twitter for $44 billion\"")
        print("   python query.py \"Climate change is a hoax\"")
        print("   python query.py \"COVID vaccines contain microchips\"")
        sys.exit(1)
    
    # Get the claim from command line arguments
    claim = " ".join(sys.argv[1:])
    
    # Run the analysis
    try:
        asyncio.run(analyze_claim(claim))
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()