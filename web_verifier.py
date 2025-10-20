from __future__ import annotations

import os
import json
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Multi-Agent Fact-Checking Functions

async def process_claim_with_multi_agent(news: str) -> Dict[str, Any]:
	"""Process a news claim using the multi-agent fact-checking system"""
	from llm_agents import MultiAgentFactChecker
	
	try:
		# Initialize the new multi-agent fact checker
		fact_checker = MultiAgentFactChecker()
		
		# Process the claim through all 6 agents
		result = await fact_checker.process_claim(news)
		
		# Convert to the expected output format for compatibility
		final_out = {
			"news": news,
			"verdict": result.final_verdict.lower() == "real",  # Convert to boolean
			"confidence": result.confidence / 100.0,  # Convert percentage to decimal
			"context": "multi_agent",  # Indicate this came from multi-agent system
			"category": result.domain,
			"bias": "none",  # Could be enhanced with bias detection from A4
			"verified_sources": [source.get("url", "") for source in result.sources],
			"entity_details": {
				"name": ", ".join(result.named_entities[:3]) if result.named_entities else "Unknown",
				"label": None,
				"summary": None,
				"url": None,
				"source": "multi_agent_analysis"
			},
			"trace": [
				{
					"source": source.get("title", "Unknown Source"),
					"snippet": source.get("snippet", ""),
					"url": source.get("url", "")
				}
				for source in result.sources[:5]  # Limit to first 5 sources
			],
			"explanation": f"Multi-agent analysis: {result.final_verdict} with {result.confidence}% confidence. " +
						  f"Found {len(result.sources)} sources. Key findings: {'; '.join(result.fact_finding[:2])}"
		}
		
		# Add detailed multi-agent specific fields
		final_out["multi_agent_details"] = {
			"domain": result.domain,
			"named_entities": result.named_entities,
			"fact_findings": result.fact_finding,
			"agent_count": 6,
			"analysis_type": "parallel_fusion"
		}
		
		return final_out
		
	except Exception as e:
		# Error handling
		return {
			"news": news,
			"error": f"Multi-agent analysis failed: {str(e)}",
			"verdict": "Uncertain",
			"confidence": 0.5,
			"context": "error",
			"category": "general",
			"bias": "none",
			"verified_sources": [],
			"entity_details": {"name": "Unknown", "label": None, "summary": None, "url": None, "source": None},
			"trace": [],
			"explanation": f"Analysis failed due to: {str(e)}"
		}


def save_to_verified_reports(final_out: Dict[str, Any], reports_path: str = "data/verified_reports.json") -> None:
	"""Save analysis results to verified reports file"""
	try:
		import os
		import json
		
		os.makedirs(os.path.dirname(reports_path), exist_ok=True)
		
		# Load existing reports
		try:
			with open(reports_path, "r", encoding="utf-8") as f:
				existing = json.load(f)
		except FileNotFoundError:
			existing = []
		
		# Add new report
		existing.append(final_out)
		
		# Save updated reports
		with open(reports_path, "w", encoding="utf-8") as f:
			json.dump(existing, f, ensure_ascii=False, indent=2)
	except Exception:
		# Don't fail if we can't save, just continue
		pass

