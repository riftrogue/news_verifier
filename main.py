#!/usr/bin/env python3
"""
Enhanced main.py using the new MultiAgentFactChecker system
Provides sophisticated fact-checking with 6-agent architecture
"""

import os
import sys
import json
import asyncio

from dotenv import load_dotenv
from web_verifier import process_claim_with_multi_agent, save_to_verified_reports


async def main():
    """Main function using the new multi-agent fact-checking system"""
    load_dotenv()
    
    # Input: either CLI arg or JSON file
    if len(sys.argv) > 1:
        news = " ".join(sys.argv[1:]).strip()
    else:
        # fallback to temp_input.json
        try:
            with open("data/temp_input.json", "r", encoding="utf-8") as f:
                tmp = json.load(f)
                news = tmp.get("news") or ""
        except Exception:
            news = ""

    if not news:
        print(json.dumps({
            "error": "No input provided. Pass as CLI args or set data/temp_input.json with key 'news'"
        }, ensure_ascii=False))
        return

    # Process the claim using multi-agent system
    result = await process_claim_with_multi_agent(news)
    
    # Save to verified reports
    save_to_verified_reports(result)
    
    # Print final JSON output
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())