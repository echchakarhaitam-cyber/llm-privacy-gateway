"""Per-token logistic regression on the CRF's features: the ablation without transitions.

Each token's BIO tag is predicted independently by a multinomial
``sklearn.linear_model.LogisticRegression`` (lbfgs, ``max_iter`` 1000, ``class_weight``
None) over the same feature dicts as the CRF, one-hot encoded by ``DictVectorizer``
(float32, sparse). Without transition weights nothing stops ``O I-PERSON`` or
``B-EMAIL I-PERSON``; :func:`privgate.ml.bio.bio_to_spans` repairs such sequences (an
orphan ``I-`` starts a span). A span's score is the lowest probability of its tokens.

The vectorizer and model are saved with ``pickle``; the sidecar's SHA-256 is checked before
unpickling (privgate.ml.model_io), and only files this project wrote should be loaded.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from privgate.data.documents import Document
from privgate.ml.bio import bio_to_spans, spans_to_bio
from privgate.ml.features import doc_features
from privgate.ml.model_io import read_sidecar, training_set_sha256, write_sidecar
from privgate.ml.tokens import tokenize
from privgate.spans import EntityType, Span

MODEL_FILE = "model.pkl"
PACKAGES = ("scikit-learn", "numpy", "scipy")


def _sklearn():
    try:
        import numpy as np
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("TokenLogRegDetector needs the ml extra: pip install -e '.[ml]'") \
            from exc
    return np, DictVectorizer, LogisticRegression


class TokenLogRegDetector:
    name = "logreg"
    supported_labels: frozenset[EntityType] = frozenset(EntityType)

    def __init__(self, vectorizer: Any, model: Any, meta: dict | None = None):
        self._vectorizer = vectorizer
        self._model = model
        self.meta = meta or {}

    @classmethod
    def train(cls, docs: Sequence[Document], C: float = 1.0, max_iter: int = 1000,
              seed: int = 0) -> TokenLogRegDetector:
        np, DictVectorizer, LogisticRegression = _sklearn()
        t0 = time.perf_counter()
        tags: list[str] = []

        def feature_stream():
            for doc in docs:
                tokens = tokenize(doc.text)
                tags.extend(spans_to_bio(tokens, doc.spans))
                yield from doc_features(doc.text, tokens)

        vectorizer = DictVectorizer(dtype=np.float32, sparse=True)
        X = vectorizer.fit_transform(feature_stream())
        model = LogisticRegression(C=C, max_iter=max_iter, class_weight=None,
                                   random_state=seed)
        model.fit(X, tags)
        meta = {
            "kind": "logreg",
            "hyperparameters": {"C": C, "max_iter": max_iter, "class_weight": None,
                                "solver": model.solver, "n_iter": int(max(model.n_iter_))},
            "training_seconds": round(time.perf_counter() - t0, 3),
            "n_docs": len(docs),
            "n_tokens": len(tags),
            "n_features": len(vectorizer.vocabulary_),
            "seed": seed,
            "training_set_sha256": training_set_sha256(docs),
        }
        return cls(vectorizer, model, meta)

    def detect(self, text: str) -> list[Span]:
        tokens = tokenize(text)
        if not tokens:
            return []
        X = self._vectorizer.transform(doc_features(text, tokens))
        proba = self._model.predict_proba(X)
        best = proba.argmax(axis=1)
        tags = [str(self._model.classes_[k]) for k in best]
        scores = [float(proba[i, k]) for i, k in enumerate(best)]
        return bio_to_spans(tokens, tags, source=self.name, scores=scores)

    def save(self, model_dir: Path) -> dict:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / MODEL_FILE).open("wb") as fh:
            pickle.dump({"vectorizer": self._vectorizer, "model": self._model}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        self.meta = write_sidecar(model_dir, MODEL_FILE, self.meta, PACKAGES)
        return self.meta

    @classmethod
    def load(cls, model_dir: Path) -> TokenLogRegDetector:
        meta = read_sidecar(model_dir, "logreg")  # verifies the file hash first
        with (Path(model_dir) / MODEL_FILE).open("rb") as fh:
            # Safe only because read_sidecar verified this project's own file hash above.
            payload = pickle.load(fh)
        return cls(payload["vectorizer"], payload["model"], meta)
