from __future__ import annotations

import os
import json
from typing import List, Dict, Optional

import yaml
import requests
from urllib.parse import quote
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


def _slugify_for_wikipedia(title: str) -> str:
	slug = title.strip().replace(" ", "_")
	return slug


def enrich_entity(entity_name: str) -> Optional[Dict]:
	"""Return a compact enrichment from Wikipedia: label/description and summary.

	Priority: find best Wikipedia page via Tavily include_domains. Fallback: call
	REST summary endpoint directly with slugified entity name (Wikipedia usually redirects).
	"""
	if not entity_name or entity_name.lower() == "unknown":
		return None

	api_key = os.getenv("TAVILY_API_KEY")
	page_url = None
	title_slug = None
	# Try Tavily first
	tavily_item = None
	if TavilySearch is not None and api_key:
		try:
			tool = TavilySearch(max_results=3, topic="general", include_answer=False, include_raw_content=False)
			out = tool.invoke({"query": entity_name, "include_domains": ["en.wikipedia.org", "wikipedia.org"], "search_depth": "basic"})
			for item in out.get("results", []):
				url = item.get("url")
				if not url:
					continue
				host = url.split("/")[2]
				if "wikipedia.org" in host:
					page_url = url
					tavily_item = item
					# extract slug after /wiki/
					try:
						idx = url.find("/wiki/")
						if idx != -1:
							title_slug = url[idx+6:]
					except Exception:
						pass
					break
		except Exception:
			pass

	# Fallback: best-effort slug
	if not title_slug:
		title_slug = _slugify_for_wikipedia(entity_name)
		page_url = page_url or f"https://en.wikipedia.org/wiki/{title_slug}"

	# Fetch REST summary
	try:
		enc_slug = quote(title_slug, safe="_")
		resp = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{enc_slug}", timeout=10, headers={"accept": "application/json"})
		if resp.status_code == 200:
			data = resp.json()
			label = data.get("description") or ""
			summary = data.get("extract") or ""
			title = data.get("title") or entity_name
			# derive a simple role from label/summary
			role = _derive_role(label, summary)
			return {
				"name": title,
				"label": label,
				"role": role,
				"summary": summary,
				"url": page_url,
				"source": "wikipedia.org",
			}
	except Exception:
		pass

	# Fallback to Tavily snippet if summary failed
	if tavily_item:
		title = tavily_item.get("title") or entity_name
		snippet = tavily_item.get("content") or ""
		role = _derive_role("", snippet)
		return {
			"name": title,
			"label": None,
			"role": role,
			"summary": snippet,
			"url": tavily_item.get("url"),
			"source": "wikipedia.org",
		}

	return None


def _derive_role(label: str, text: str) -> Optional[str]:
	s = f"{label} {text}".lower()
	role_keywords = [
		"actor", "actress", "boxer", "cricketer", "footballer", "politician", "singer",
		"director", "producer", "businessman", "entrepreneur", "journalist", "author",
		"scientist", "youtuber", "influencer", "athlete"
	]
	for kw in role_keywords:
		if kw in s:
			return "actor" if kw == "actress" else kw
	return None

