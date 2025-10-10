from __future__ import annotations

import os
import json
from typing import List, Dict, Optional

import yaml
import requests
from dotenv import load_dotenv

try:
	from langchain_tavily import TavilySearch
except Exception:  # Allow import failure until deps installed
	TavilySearch = None  # type: ignore


load_dotenv()


def load_settings(path: str = "configs/settings.yaml") -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def _fetch_title(url: str, timeout: int = 10) -> Optional[str]:
	try:
		resp = requests.get(url, timeout=timeout)
		if resp.status_code != 200:
			return None
		# naive title extraction
		text = resp.text
		start = text.lower().find("<title>")
		end = text.lower().find("</title>")
		if start != -1 and end != -1 and end > start:
			return text[start + 7 : end].strip()
		return None
	except Exception:
		return None


def verify_with_trusted_sources(query: str, include_domains: List[str]) -> List[Dict]:
	"""Try to query trusted domains via Tavily include_domains filter if available.
	Fallback: naive requests to homepage search endpoints is skipped; we rely on Tavily.
	"""
	results: List[Dict] = []
	api_key = os.getenv("TAVILY_API_KEY")
	if TavilySearch is None or not api_key:
		return results

	tool = TavilySearch(max_results=5, topic="news", include_answer=False, include_raw_content=False)
	for domain in include_domains:
		try:
			out = tool.invoke({"query": query, "include_domains": [domain], "search_depth": "basic"})
			for item in out.get("results", []):
				results.append({
					"title": item.get("title"),
					"url": item.get("url"),
					"source": domain,
				})
		except Exception:
			continue
	return results


def verify_with_tavily(query: str, max_results: int = 5, include_domains: Optional[List[str]] = None) -> List[Dict]:
	api_key = os.getenv("TAVILY_API_KEY")
	if TavilySearch is None or not api_key:
		return []
	tool = TavilySearch(max_results=max_results, topic="news", include_answer=False, include_raw_content=False)
	try:
		out = tool.invoke({"query": query, "search_depth": "basic", "include_domains": include_domains})
	except Exception:
		return []

	results: List[Dict] = []
	for item in out.get("results", []):
		url = item.get("url")
		results.append({
			"title": item.get("title") or _fetch_title(url) or "",
			"url": url,
			"source": url.split("/")[2] if url else "",
		})
	return results


def web_verify(query: str, settings_path: str = "configs/settings.yaml") -> List[Dict]:
	settings = load_settings(settings_path)
	trusted = settings.get("trusted_sources", [])
	ver_cfg = settings.get("verification", {})
	max_results = int(ver_cfg.get("max_results", 5))

	# Try trusted sources first
	trusted_hits = verify_with_trusted_sources(query, trusted)
	if trusted_hits:
		return trusted_hits[:max_results]

	# Fallback to Tavily general
	return verify_with_tavily(query, max_results=max_results)

