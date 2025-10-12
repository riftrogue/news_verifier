from __future__ import annotations

import os
import sys
import json
from typing import List, Dict

from dotenv import load_dotenv
import yaml

from llm_agents import LLMAgentA, LLMAgentB, LLMAgentC, _load_json
from vector_db import SimpleVectorDB
from web_verifier import web_verify, enrich_entity


def load_settings(path: str = "configs/settings.yaml") -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}

## Trace execution , will remove later
def _trace(settings: dict, stage: str, payload: dict):
	dbg = settings.get("debug", {})
	if not dbg.get("log_trace", False):
		return
	trace_path = dbg.get("trace_path", "data/run_trace.jsonl")
	try:
		os.makedirs(os.path.dirname(trace_path), exist_ok=True)
		with open(trace_path, "a", encoding="utf-8") as f:
			f.write(json.dumps({"stage": stage, **payload}, ensure_ascii=False) + "\n")
	except Exception:
		pass


def _clean_entity_name(name: str) -> str:
	if not name:
		return name
	s = name.strip()
	lower = s.lower()
	# cut phrases like " from <...>" or following commas/pipe decorations
	for token in [" from ", ",", " | "]:
		if token in s:
			idx = lower.find(token.strip()) if token.strip() in ["from"] else s.find(token)
			if idx != -1:
				s = s[:idx].strip()
				lower = s.lower()
	# remove trailing " is ..."
	if " is " in lower:
		s = s[: lower.find(" is ")].strip()
	return s


def detect_context(query: str) -> str:
	q = query.lower()
	if "born" in q or "birthplace" in q:
		return "birthplace"
	if any(k in q for k in ["cricketer", "actor", "politician", "singer", "works as", "is a "]):
		return "profession"
	if any(k in q for k in [" is in ", " located in ", " is located in "]):
		return "location"
	return "general"


def _extract_subject_object_for_context(query: str) -> Dict:
	"""Very lightweight subject/object extraction for simple patterns.
	Returns {subject, object} where subject is likely the entity and object is the location/profession.
	"""
	q = query.strip()
	lower = q.lower()
	subj = None
	obj = None
	# Pattern: "X was born in Y" / "X born in Y"
	for key in [" was born in ", " born in "]:
		if key in lower:
			i = lower.find(key)
			subj = q[:i].strip().strip(". ?!")
			obj = q[i+len(key):].strip().strip(". ?!")
			break
	# Pattern: "X is a Y" or "X works as Y"
	if subj is None:
		for key in [" is a ", " works as "]:
			if key in lower:
				i = lower.find(key)
				subj = q[:i].strip().strip(". ?!")
				obj = q[i+len(key):].strip().strip(". ?!")
				break
	# Pattern: "X is in Y" / "X located in Y"
	if subj is None:
		for key in [" is in ", " located in ", " is located in "]:
			if key in lower:
				i = lower.find(key)
				subj = q[:i].strip().strip(". ?!")
				obj = q[i+len(key):].strip().strip(". ?!")
				break
	return {"subject": subj, "object": obj}




def main():
	load_dotenv()
	settings = load_settings()
	chat_history_path = settings.get("storage", {}).get("chat_history_path", "data/chat_history.json")
	vector_index_path = settings.get("storage", {}).get("vector_index_path", "data/embeddings")
	verified_reports_path = settings.get("storage", {}).get("verified_reports_path", "data/verified_reports.json")

	# Input: either CLI arg or JSON file
	if len(sys.argv) > 1:
		news = " ".join(sys.argv[1:]).strip()
	else:
		# fallback to temp_input.json
		try:
			tmp = _load_json("data/temp_input.json", {})
			news = tmp.get("news") or ""
		except Exception:
			news = ""

	if not news:
		print(json.dumps({"error": "No input provided. Pass as CLI args or set data/temp_input.json with key 'news'"}))
		return

	# LLM_A: analyze silently (logs chat history)
	agent_a = LLMAgentA(chat_history_path)
	a_info = agent_a.analyze(news)
	# Clean entity name to reduce ambiguity for search/enrichment
	if a_info.get("entity"):
		a_info["entity"] = _clean_entity_name(a_info["entity"])
	# Detect context and parse subject/object for rule checks
	context = detect_context(news)
	claim_parts = _extract_subject_object_for_context(news)
	_trace(settings, "LLM_A", {"a_info": a_info})

	# VectorDB search
	vectordb = SimpleVectorDB(index_dir=vector_index_path)
	vec_hits = []
	try:
		vec_hits_raw = vectordb.search(news, k=5, score_threshold=0.6)
		for r in vec_hits_raw:
			vec_hits.append({"text": r.text, "score": r.score, "metadata": r.metadata, "source": (r.metadata or {}).get("source")})
	except Exception:
		vec_hits = []
	_trace(settings, "VECTOR_SEARCH", {"hits": vec_hits})

	evidence: List[Dict] = []
	evidence.extend(vec_hits)

	# If vector search is empty, verify on web
	if not vec_hits:
		web_hits = web_verify(news, category=a_info.get("category"), entity=a_info.get("entity"))
		# shape into evidence format
		for it in web_hits:
			evidence.append({"text": it.get("title"), "source": it.get("source"), "url": it.get("url")})
		_trace(settings, "WEB_VERIFY", {"hits": web_hits})

	# Load chat history for B
	chat_history = _load_json(chat_history_path, [])

	# LLM_B analytical summary
	agent_b = LLMAgentB()
	b_info = agent_b.analyze(news, evidence, chat_history)
	_trace(settings, "LLM_B", {"b_info": b_info})

	# LLM_C final verdict
	tone = a_info.get("sentiment") or "none"
	agent_c = LLMAgentC()
	base_conf = float(b_info.get("confidence", 0.5))
	final = agent_c.summarize(news, b_info.get("analysis", ""), base_conf, tone)
	_trace(settings, "LLM_C", {"final_raw": final})

	# augment with required fields
	# Optional entity enrichment
	enrichment = None
	entity_name = a_info.get("entity")
	try:
		enrichment = enrich_entity(entity_name) if entity_name else None
	except Exception:
		enrichment = None
	_trace(settings, "ENRICH_ENTITY", {"entity": entity_name, "enrichment": enrichment})

	# Confidence calibration and context-aware adjustment
	adjusted_conf = float(final.get("confidence", base_conf))
	verdict_val = final.get("verdict")
	v_true = (str(verdict_val).lower() == "true") or (verdict_val is True)

	# Pull a summary text and structured fields for rule checks
	summary_text = (enrichment or {}).get("summary") or ""
	birthplace = (enrichment or {}).get("birthplace")
	occupations = (enrichment or {}).get("occupations") or []
	subject = (claim_parts.get("subject") or entity_name or "").lower()
	obj = (claim_parts.get("object") or "").lower()

	# Rule-based nudges
	if context == "birthplace":
		# Prefer structured birthplace from Wikidata, fallback to summary text heuristic
		if obj:
			if birthplace and obj in birthplace.lower():
				adjusted_conf = min(1.0, adjusted_conf + 0.25)
				v_true = True
			elif ("born in " + obj) in summary_text.lower() or ("birthplace" in summary_text.lower() and obj in summary_text.lower()):
				adjusted_conf = min(1.0, adjusted_conf + 0.15)
				v_true = True
	elif context == "profession":
		# Prefer structured occupations, fallback to summary text
		if obj:
			if any(obj in (occ or "").lower() for occ in occupations):
				adjusted_conf = min(1.0, adjusted_conf + 0.25)
				v_true = True
			elif obj in summary_text.lower():
				adjusted_conf = min(1.0, adjusted_conf + 0.15)
				v_true = True
	elif context == "location":
		if obj and obj in summary_text.lower():
			adjusted_conf = min(1.0, adjusted_conf + 0.2)
			v_true = True

	# Trusted sources boost/penalty (consider both evidence and enrichment source)
	trusted = set(settings.get("trusted_sources", []))
	srcs = [ev.get("source") for ev in evidence if ev.get("source")]
	if (enrichment or {}).get("source"):
		srcs.append((enrichment or {}).get("source"))
	has_trusted = any(s in trusted for s in srcs)
	if has_trusted:
		adjusted_conf = min(1.0, adjusted_conf + 0.1)
	else:
		adjusted_conf = max(0.0, adjusted_conf - 0.2)

	# Perfect entity name match boost
	if entity_name and entity_name.lower() in (enrichment or {}).get("name", "").lower():
		adjusted_conf = min(1.0, adjusted_conf + 0.1)

	final["confidence"] = float(f"{adjusted_conf:.3f}")
	final_verdict = True if v_true else False if str(final.get("verdict")).lower() == "false" or final.get("verdict") is False else final.get("verdict")

	# Collate verified sources, including enrichment source if present
	verified_sources = {ev.get("source") for ev in evidence if ev.get("source")}
	if (enrichment or {}).get("source"):
		verified_sources.add((enrichment or {}).get("source"))

	final_out = {
		"news": news,
		"verdict": final_verdict if final_verdict in [True, False] else final.get("verdict", "Uncertain"),
		"confidence": final.get("confidence", b_info.get("confidence", 0.5)),
		"context": context,
		"bias": final.get("bias", tone or "none"),
		"verified_sources": list(verified_sources),
		"entity_details": enrichment or {"name": entity_name, "label": None, "summary": None, "url": None, "source": None},
	}

	# Persist lightweight report
	try:
		os.makedirs(os.path.dirname(verified_reports_path), exist_ok=True)
		existing = _load_json(verified_reports_path, [])
		existing.append(final_out)
		with open(verified_reports_path, "w", encoding="utf-8") as f:
			json.dump(existing, f, ensure_ascii=False, indent=2)
	except Exception:
		pass

	# Print final JSON only
	print(json.dumps(final_out, ensure_ascii=False))


if __name__ == "__main__":
	main()

