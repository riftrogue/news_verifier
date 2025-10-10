from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()


def _load_json(path: str, default):
	try:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return default


def _save_json(path: str, data):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


def get_llm(model: Optional[str] = None) -> ChatGroq:
	# Prefer env override; fallback to a supported public Groq model.
	# You can set GROQ_MODEL in .env, e.g., GROQ_MODEL="deepseek-r1-distill-llama-70b"
	model_name = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
	return ChatGroq(model=model_name, temperature=0.1)


class LLMAgentA:
	"""Input analyzer: extracts category, key entity/topic, sentiment.
	Stores conversational-style logs in chat_history.json
	"""

	def __init__(self, chat_history_path: str):
		self.chat_history_path = chat_history_path
		self.llm = get_llm()
		self.prompt = ChatPromptTemplate.from_messages([
			("system", "You analyze a news claim and return a concise line: 'Category: <category> | Key Entity: <entity/topic> | Sentiment: <Neutral/Positive/Negative>'. Keep it short."),
			("human", "News: {news}")
		])

	def analyze(self, news: str) -> Dict:
		chain = self.prompt | self.llm
		resp = chain.invoke({"news": news})
		content = resp.content.strip()
		# log chat style
		history = _load_json(self.chat_history_path, [])
		history.append({"role": "user", "content": news})
		history.append({"role": "assistant", "content": content})
		_save_json(self.chat_history_path, history)

		# light parse for downstream use
		category = "unknown"
		entity = "unknown"
		sentiment = "unknown"
		try:
			parts = [p.strip() for p in content.split("|")]
			for p in parts:
				if p.lower().startswith("category:"):
					category = p.split(":", 1)[1].strip()
				elif p.lower().startswith("key entity:") or p.lower().startswith("key person:") or p.lower().startswith("key topic:"):
					entity = p.split(":", 1)[1].strip()
				elif p.lower().startswith("sentiment:"):
					sentiment = p.split(":", 1)[1].strip()
		except Exception:
			pass
		return {"category": category, "entity": entity, "sentiment": sentiment, "raw": content}


class LLMAgentB:
	"""Analytical agent: uses retrieved evidence and chat history to produce contradiction analysis and confidence."""

	def __init__(self):
		self.llm = get_llm()
		self.prompt = ChatPromptTemplate.from_messages([
			("system", "You are a fact-checking analyst. Given a news claim, relevant evidence snippets (from vector DB or the web), and prior chat context, produce: (1) brief contradiction/support analysis, (2) short evidence citations by source names, and (3) a numeric confidence 0-1. Keep under 120 words."),
			("human", "Claim: {news}\nEvidence:\n{evidence}\nChat History (may be partial): {chat_history}")
		])

	def analyze(self, news: str, evidence_snippets: List[Dict], chat_history: List[Dict]) -> Dict:
		ev_lines = []
		for ev in evidence_snippets:
			src = ev.get("source") or (ev.get("metadata") or {}).get("source") or "unknown"
			txt = ev.get("text") or ev.get("title") or ev.get("raw") or ""
			ev_lines.append(f"- [{src}] {txt}")
		evidence_text = "\n".join(ev_lines) if ev_lines else "(none)"
		chain = self.prompt | self.llm
		resp = chain.invoke({
			"news": news,
			"evidence": evidence_text,
			"chat_history": json.dumps(chat_history[-6:], ensure_ascii=False)
		})
		content = resp.content.strip()
		# naive confidence extraction e.g., 'confidence: 0.78'
		conf = 0.5
		for token in content.lower().split():
			try:
				val = float(token)
				if 0.0 <= val <= 1.0:
					conf = val
					break
			except Exception:
				continue
		return {"analysis": content, "confidence": conf}


class LLMAgentC:
	"""Bias detector and final verdict summarizer."""

	def __init__(self):
		self.llm = get_llm()
		self.prompt = ChatPromptTemplate.from_messages([
			("system", "You are a bias detector and final summarizer. Given a claim, an analytical summary with confidence, and any detected tone, produce a final JSON object with keys: verdict (True/False/Uncertain), confidence (0-1), bias (e.g., none/leaning/inflammatory), and a 1-sentence justification. Keep it concise and JSON-valid only."),
			("human", "Claim: {news}\nAnalysis: {analysis}\nConfidence: {confidence}\nTone: {tone}")
		])

	def summarize(self, news: str, analysis: str, confidence: float, tone: str) -> Dict:
		chain = self.prompt | self.llm
		resp = chain.invoke({
			"news": news,
			"analysis": analysis,
			"confidence": confidence,
			"tone": tone,
		})
		content = resp.content.strip()
		try:
			data = json.loads(content)
		except Exception:
			# fallback structure
			data = {"verdict": "Uncertain", "confidence": confidence, "bias": tone or "none", "justification": content[:200]}
		return data

