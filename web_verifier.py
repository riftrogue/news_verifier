from __future__ import annotations

import os
import json
from typing import List, Dict, Optional, Tuple

import yaml
import requests
from urllib.parse import quote
from dotenv import load_dotenv

try:
	from langchain_tavily import TavilySearch
except Exception:  # Allow import failure until deps installed
	TavilySearch = None  # type: ignore


load_dotenv()

# Global User-Agent for Wikipedia/Wikidata requests to avoid 403s and improve reliability
UA = {"User-Agent": "news_verifier/1.0 (+https://github.com/)"}


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

	tool = TavilySearch(max_results=8, topic="news", include_answer=False, include_raw_content=False)
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


def _build_queries(news: str, entity: Optional[str], category: Optional[str], official_site_keywords: List[str]) -> List[str]:
	queries: List[str] = []
	if entity:
		queries.append(entity)
		if category:
			queries.append(f"{entity} {category}")
		for kw in official_site_keywords:
			queries.append(f"{entity} {kw}")
	# Also add the full news text as a fallback
	if news:
		queries.append(news)
	# Dedup while preserving order
	seen = set()
	out: List[str] = []
	for q in queries:
		if q and q not in seen:
			seen.add(q)
			out.append(q)
	return out


def web_verify(query: str, settings_path: str = "configs/settings.yaml", category: Optional[str] = None, entity: Optional[str] = None) -> List[Dict]:
	settings = load_settings(settings_path)
	trusted = settings.get("trusted_sources", [])
	trusted_general = settings.get("trusted_general", [])
	trusted_by_category = settings.get("trusted_by_category", {})
	social_domains = settings.get("social_domains", [])
	official_kw = settings.get("official_site_keywords", [])
	ver_cfg = settings.get("verification", {})
	max_results = int(ver_cfg.get("max_results", 5))

	# Build prioritized domain list: category-specific -> social (official pages) -> general -> global trusted
	include_domains = []
	if category and isinstance(trusted_by_category.get(category.lower()), list):
		include_domains.extend(trusted_by_category[category.lower()])
	include_domains.extend(social_domains)
	include_domains.extend(trusted_general)
	include_domains.extend([d for d in trusted if d not in include_domains])

	# Build queries and search across them for trusted/category domains
	queries = _build_queries(query, entity, category, official_kw)
	collected: List[Dict] = []
	for q in queries:
		hits = verify_with_trusted_sources(q, include_domains)
		if hits:
			collected.extend(hits)
		# stop early if sufficient results
		if len(collected) >= max_results * 2:
			break
	if collected:
		# de-duplicate by URL and keep order
		seen = set()
		dedup = []
		for it in collected:
			u = it.get("url")
			if not u or u in seen:
				continue
			dedup.append(it)
			seen.add(u)
		return dedup[:max_results]

	# Fallback to Tavily general
	return verify_with_tavily(query, max_results=max_results)


def _slugify_for_wikipedia(title: str) -> str:
	slug = title.strip().replace(" ", "_")
	return slug


def _norm_name(s: str) -> str:
	return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


def _name_similarity(a: str, b: str) -> float:
	# Cheap token overlap similarity (0-1)
	ta = set(_norm_name(a).split())
	tb = set(_norm_name(b).split())
	if not ta or not tb:
		return 0.0
	inter = len(ta & tb)
	uni = len(ta | tb)
	return inter / uni


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
	best_score = 0.0
	if TavilySearch is not None and api_key:
		try:
			tool = TavilySearch(max_results=5, topic="general", include_answer=False, include_raw_content=False)
			out = tool.invoke({"query": entity_name, "include_domains": ["en.wikipedia.org", "wikipedia.org"], "search_depth": "basic"})
			for item in out.get("results", []):
				url = item.get("url")
				if not url:
					continue
				host = url.split("/")[2]
				if "wikipedia.org" in host:
					title = item.get("title") or ""
					# skip disambiguation pages
					if "disambiguation" in title.lower():
						continue
					sim = _name_similarity(entity_name, title)
					# prefer English Wikipedia host
					host_bonus = 0.2 if host.startswith("en.wikipedia.org") else 0.0
					score = sim + host_bonus
					if score >= best_score:
						best_score = score
						page_url = url
						tavily_item = item
					# extract slug after /wiki/
					try:
						idx = url.find("/wiki/")
						if idx != -1:
							title_slug = url[idx+6:]
					except Exception:
						pass
		except Exception:
			pass

	# Fallback: best-effort slug
	if not title_slug:
		title_slug = _slugify_for_wikipedia(entity_name)
		page_url = page_url or f"https://en.wikipedia.org/wiki/{title_slug}"

	# Fetch REST summary
	try:
		enc_slug = quote(title_slug, safe="_")
		resp = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{enc_slug}", timeout=10, headers={"accept": "application/json", **UA})
		if resp.status_code == 200:
			data = resp.json()
			# avoid disambiguation summary
			if str(data.get("type", "")).lower() == "disambiguation":
				return None
			label = data.get("description") or ""
			summary = data.get("extract") or ""
			title = data.get("title") or entity_name
			# If the resolved title is a weak match to the entity, retry with exact slug of entity_name
			try:
				if entity_name and _name_similarity(entity_name, title) < 0.8:
					alt_slug = quote(_slugify_for_wikipedia(entity_name), safe="_")
					resp_alt = requests.get(
						f"https://en.wikipedia.org/api/rest_v1/page/summary/{alt_slug}",
						timeout=10,
						headers={"accept": "application/json", **UA},
					)
					if resp_alt.status_code == 200:
						data2 = resp_alt.json()
						if str(data2.get("type", "")).lower() != "disambiguation":
							cand_title = data2.get("title") or title
							if _name_similarity(entity_name, cand_title) > _name_similarity(entity_name, title):
								title_slug = _slugify_for_wikipedia(entity_name)
								page_url = f"https://en.wikipedia.org/wiki/{title_slug}"
								label = data2.get("description") or label
								summary = data2.get("extract") or summary
								title = cand_title
			except Exception:
				pass

			# derive a simple role from label/summary
			role = _derive_role(label, summary)
			# Try to enrich with structured birthplace and occupations via Wikidata
			birthplace, occupations = _enrich_from_wikidata(title)
			return {
				"name": title,
				"label": label,
				"role": role,
				"summary": summary,
				"url": page_url,
				"source": "wikipedia.org",
				"birthplace": birthplace,
				"occupations": occupations,
			}
	except Exception:
		pass

	# Fallback to Tavily snippet if summary failed
	if tavily_item and best_score >= 0.5:
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
			"birthplace": None,
			"occupations": [],
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


def _enrich_from_wikidata(title: str) -> Tuple[Optional[str], List[str]]:
	"""Fetch birthplace (P19) and occupations (P106) using Wikidata, given an article title.

	Returns (birthplace_label_or_None, [occupation_labels]).
	"""
	try:
		# Step 1: Find the Wikidata QID via Wikipedia API
		params = {
			"action": "query",
			"prop": "pageprops",
			"titles": title,
			"format": "json",
		}
		resp = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=10, headers=UA)
		if resp.status_code != 200:
			return None, []
		data = resp.json()
		pages = data.get("query", {}).get("pages", {})
		wikidata_id = None
		for _pid, page in pages.items():
			pp = page.get("pageprops", {})
			wikidata_id = pp.get("wikibase_item")
			if wikidata_id:
				break
		if not wikidata_id:
			# Fallback: use Wikidata search by title label
			resp_search = requests.get(
				"https://www.wikidata.org/w/api.php",
				params={
					"action": "wbsearchentities",
					"search": title,
					"language": "en",
					"format": "json",
					"limit": 1,
					"type": "item",
				},
				timeout=10,
			)
			if resp_search.status_code == 200:
				hits = resp_search.json().get("search", [])
				if hits:
					wikidata_id = hits[0].get("id")
			if not wikidata_id:
				return None, []

		# Step 2: Get claims for P19 (place of birth) and P106 (occupation)
		resp2 = requests.get(
			"https://www.wikidata.org/w/api.php",
			params={
				"action": "wbgetentities",
				"ids": wikidata_id,
				"format": "json",
				"props": "claims",
			},
			timeout=10,
			headers=UA,
		)
		if resp2.status_code != 200:
			return None, []
		ent = resp2.json().get("entities", {}).get(wikidata_id, {})
		claims = ent.get("claims", {})
		def _extract_entity_ids(prop: str) -> List[str]:
			out: List[str] = []
			for cl in claims.get(prop, []) or []:
				m = cl.get("mainsnak", {})
				datav = m.get("datavalue", {})
				val = datav.get("value", {})
				if isinstance(val, dict) and val.get("entity-type") == "item" and val.get("id"):
					out.append(val.get("id"))
			return out

		birth_ids = _extract_entity_ids("P19")
		occ_ids = _extract_entity_ids("P106")
		labels: Dict[str, str] = {}
		all_ids = birth_ids + occ_ids
		if all_ids:
			resp3 = requests.get(
				"https://www.wikidata.org/w/api.php",
				params={
					"action": "wbgetentities",
					"ids": "|".join(all_ids),
					"format": "json",
					"props": "labels",
					"languages": "en",
				},
				timeout=10,
				headers=UA,
			)
			if resp3.status_code == 200:
				eds = resp3.json().get("entities", {})
				for qid, meta in eds.items():
					lab = (meta.get("labels", {}).get("en", {}) or {}).get("value")
					if lab:
						labels[qid] = lab
		birthplace = labels.get(birth_ids[0]) if birth_ids else None
		occupations = [labels[q] for q in occ_ids if q in labels]
		return birthplace, occupations
	except Exception:
		return None, []

