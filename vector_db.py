from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
	import faiss  # type: ignore
except Exception as e:  # pragma: no cover
	faiss = None  # Allow import on systems without faiss until install

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@dataclass
class SearchResult:
	text: str
	score: float
	metadata: Optional[dict] = None


class SimpleVectorDB:
	"""A minimal FAISS-backed vector DB for text passages.

	- Stores texts and optional metadata in a JSON sidecar.
	- Stores FAISS index in a binary file.
	- Embedding model via sentence-transformers.
	"""

	def __init__(self, index_dir: str, model_name: str = DEFAULT_MODEL):
		self.index_dir = index_dir
		os.makedirs(index_dir, exist_ok=True)
		self.index_path = os.path.join(index_dir, "index.faiss")
		self.meta_path = os.path.join(index_dir, "meta.json")
		self._model_name = model_name
		self._model: Optional[SentenceTransformer] = None
		self.index = None
		self.texts: List[str] = []
		self.metadatas: List[Optional[dict]] = []
		# defer model load to avoid heavy startup
		self.dim: Optional[int] = None

		if os.path.exists(self.index_path) and os.path.exists(self.meta_path) and faiss is not None:
			self._load()

	def _persist(self):
		if faiss is None or self.index is None:
			return
		faiss.write_index(self.index, self.index_path)
		with open(self.meta_path, "w", encoding="utf-8") as f:
			json.dump({"texts": self.texts, "metadatas": self.metadatas}, f, ensure_ascii=False, indent=2)

	def _load(self):
		if faiss is None:
			return
		self.index = faiss.read_index(self.index_path)
		with open(self.meta_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		self.texts = data.get("texts", [])
		self.metadatas = data.get("metadatas", [None] * len(self.texts))

	def _embed(self, texts: List[str]) -> np.ndarray:
		if self._model is None:
			self._model = SentenceTransformer(self._model_name)
			if self.dim is None:
				self.dim = self._model.get_sentence_embedding_dimension()
			if self.index is None and faiss is not None:
				self.index = faiss.IndexFlatIP(self.dim)
		embs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
		return embs.astype("float32")

	def add(self, texts: List[str], metadatas: Optional[List[Optional[dict]]] = None):
		if faiss is None:
			raise RuntimeError("faiss is not installed. Please install faiss-cpu.")
		metadatas = metadatas or [None] * len(texts)
		self.texts.extend(texts)
		self.metadatas.extend(metadatas)
		vectors = self._embed(texts)
		self.index.add(vectors)
		self._persist()

	def search(self, query: str, k: int = 5, score_threshold: float = 0.6) -> List[SearchResult]:
		if faiss is None or self.index is None or len(self.texts) == 0:
			return []

		q = self._embed([query])
		scores, idxs = self.index.search(q, min(k, len(self.texts)))
		results: List[SearchResult] = []
		for score, idx in zip(scores[0], idxs[0]):
			if idx == -1:
				continue
			if float(score) < score_threshold:
				continue
			results.append(
				SearchResult(text=self.texts[idx], score=float(score), metadata=self.metadatas[idx])
			)
		return results

