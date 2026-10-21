"""Linear-chain CRF detector (sklearn-crfsuite / CRFsuite), see docs/theory/02-crf.md.

Training: L-BFGS on the L1 (``c1``) + L2 (``c2``) regularised negative log-likelihood,
``max_iterations`` 100, ``all_possible_transitions=True`` (weights for every tag pair,
including pairs never seen in training, so the model can learn that e.g. ``O -> I-PERSON``
is bad). Documents are streamed into CRFsuite one at a time, so the Python feature dicts of
the whole training set are never held in memory. CRFsuite's L-BFGS has no randomness: the
same documents and hyper-parameters give the same model.

Prediction: tokenise, extract features, Viterbi-decode the tag sequence, convert with
:func:`privgate.ml.bio.bio_to_spans` (spans are non-overlapping by construction).
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from privgate.data.documents import Document
from privgate.ml.bio import bio_to_spans, spans_to_bio
from privgate.ml.features import doc_features
from privgate.ml.model_io import read_sidecar, training_set_sha256, write_sidecar
from privgate.ml.tokens import tokenize
from privgate.spans import EntityType, Span

MODEL_FILE = "model.crfsuite"
PACKAGES = ("sklearn-crfsuite", "python-crfsuite")


def _crf_class():
    try:
        import sklearn_crfsuite
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("CRFDetector needs the ml extra: pip install -e '.[ml]'") from exc
    return sklearn_crfsuite.CRF


def training_sequences(docs: Iterable[Document]) -> Iterator[tuple[list[dict], list[str]]]:
    """(features, BIO tags) per document, lazily."""
    for doc in docs:
        tokens = tokenize(doc.text)
        yield doc_features(doc.text, tokens), spans_to_bio(tokens, doc.spans)


def split_stream(pairs: Iterable[tuple]) -> tuple[Iterator, Iterator]:
    """Two lazy streams (xs, ys) from one stream of pairs, for ``zip(xs, ys)`` consumers.

    ``zip`` pulls from ``xs`` first, which stashes the matching ``y``; ``ys`` then hands it
    over. Only one pair is alive at a time.
    """
    it = iter(pairs)
    stash: list = []

    def xs():
        for x, y in it:
            stash.append(y)
            yield x

    return xs(), _Stash(stash)


class _Stash:
    """Iterator handing over the ``y`` stashed by the matching ``x``."""

    def __init__(self, stash: list):
        self._stash = stash

    def __iter__(self):
        return self

    def __next__(self):
        if not self._stash:
            raise StopIteration
        return self._stash.pop()


class CRFDetector:
    name = "crf"
    supported_labels: frozenset[EntityType] = frozenset(EntityType)

    def __init__(self, model: Any, meta: dict | None = None):
        self._model = model
        self.meta = meta or {}

    @classmethod
    def train(cls, docs: Sequence[Document], c1: float = 0.1, c2: float = 0.1,
              max_iterations: int = 100, seed: int | None = None) -> CRFDetector:
        model = _crf_class()(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations,
                             all_possible_transitions=True)
        t0 = time.perf_counter()
        xs, ys = split_stream(training_sequences(docs))
        model.fit(xs, ys)
        meta = {
            "kind": "crf",
            "hyperparameters": {"algorithm": "lbfgs", "c1": c1, "c2": c2,
                                "max_iterations": max_iterations,
                                "all_possible_transitions": True},
            "training_seconds": round(time.perf_counter() - t0, 3),
            "n_docs": len(docs),
            "seed": seed,
            "training_set_sha256": training_set_sha256(docs),
        }
        return cls(model, meta)

    def detect(self, text: str) -> list[Span]:
        tokens = tokenize(text)
        if not tokens:
            return []
        tags = self._model.predict_single(doc_features(text, tokens))
        return bio_to_spans(tokens, tags, source=self.name)

    def transition_weights(self) -> dict[tuple[str, str], float]:
        return dict(self._model.transition_features_)

    def state_weights(self) -> dict[tuple[str, str], float]:
        return dict(self._model.state_features_)

    def save(self, model_dir: Path) -> dict:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._model.modelfile.name, model_dir / MODEL_FILE)
        self.meta = write_sidecar(model_dir, MODEL_FILE, self.meta, PACKAGES)
        return self.meta

    @classmethod
    def load(cls, model_dir: Path) -> CRFDetector:
        meta = read_sidecar(model_dir, "crf")
        model = _crf_class()(model_filename=str(Path(model_dir) / MODEL_FILE))
        return cls(model, meta)
