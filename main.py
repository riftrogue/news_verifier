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
		web_hits = web_verify(news)
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
	final = agent_c.summarize(news, b_info.get("analysis", ""), float(b_info.get("confidence", 0.5)), tone)
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

	final_out = {
		"news": news,
		"verdict": final.get("verdict", "Uncertain"),
		"confidence": final.get("confidence", b_info.get("confidence", 0.5)),
		"bias": final.get("bias", tone or "none"),
		"verified_sources": list({ev.get("source") for ev in evidence if ev.get("source")}),
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

