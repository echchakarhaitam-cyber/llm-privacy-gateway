"""Tiny CRF and per-token LogReg training, save/load round trips, model-file checks."""

import json

import pytest

from privgate.data.documents import Document
from privgate.ml.crf import split_stream
from privgate.ml.features import FEATURE_VERSION
from privgate.ml.model_io import ModelFileError, training_set_sha256
from privgate.spans import EntityType, Span

E = EntityType


NAMES = ["Albers", "Bianchi", "Castro", "Dumont", "Eriksen", "Fischer", "Garcia", "Haddad",
         "Ibsen", "Jansen", "Kowalski", "Laurent", "Moreau", "Nakamura", "Okafor"]


def synthetic_docs(n=30):
    docs = []
    for i in range(n):
        name = NAMES[i % len(NAMES)]
        text = f"Today Dr. {name} met the team; call 555-01{i:02d} or mail x{i}@clinic.org."
        start = text.index(name)
        spans = (Span(start, start + len(name), E.PERSON),)
        docs.append(Document(f"syn:{i:03d}", text, spans, "synthetic"))
    return docs


def found(det, text):
    return [(text[s.start:s.end], s.label) for s in det.detect(text)]


def test_split_stream_pairs_in_lockstep():
    xs, ys = split_stream((i, i * 10) for i in range(5))
    assert list(zip(xs, ys)) == [(i, i * 10) for i in range(5)]


@pytest.fixture(scope="module")
def crf():
    pytest.importorskip("sklearn_crfsuite")
    from privgate.ml.crf import CRFDetector

    return CRFDetector.train(synthetic_docs(), c1=0.01, c2=0.01, max_iterations=60, seed=0)


@pytest.fixture(scope="module")
def logreg():
    pytest.importorskip("sklearn")
    from privgate.ml.logreg import TokenLogRegDetector

    return TokenLogRegDetector.train(synthetic_docs(), C=10.0, max_iter=500, seed=0)


UNSEEN = "Yesterday Dr. Zorbel met us."


def test_crf_learns_title_pattern(crf):
    assert ("Zorbel", E.PERSON) in found(crf, UNSEEN)
    assert crf.detect("") == []
    assert crf.supported_labels == frozenset(EntityType)
    trans = crf.transition_weights()  # labels never seen in training (I-PERSON) get none
    assert ("O", "B-PERSON") in trans and ("B-PERSON", "O") in trans


def test_logreg_learns_title_pattern(logreg):
    assert ("Zorbel", E.PERSON) in found(logreg, UNSEEN)
    assert all(0.0 < s.score <= 1.0 for s in logreg.detect(UNSEEN))
    assert logreg.meta["n_tokens"] > 0 and logreg.meta["n_features"] > 0


@pytest.mark.parametrize("which", ["crf", "logreg"])
def test_save_load_gives_identical_predictions(which, crf, logreg, tmp_path):
    det = crf if which == "crf" else logreg
    det.save(tmp_path / which)
    meta = json.loads((tmp_path / which / "model.json").read_text())
    assert meta["kind"] == which and meta["n_docs"] == 30 and meta["seed"] == 0
    assert meta["training_set_sha256"] == training_set_sha256(synthetic_docs())
    assert meta["feature_version"] == FEATURE_VERSION and meta["training_seconds"] >= 0
    loaded = type(det).load(tmp_path / which)
    for doc in synthetic_docs()[:10] + [Document("u", UNSEEN, (), "s")]:
        assert loaded.detect(doc.text) == det.detect(doc.text)


def test_tampered_or_wrong_model_is_refused(crf, tmp_path):
    from privgate.ml.crf import CRFDetector
    from privgate.ml.logreg import TokenLogRegDetector

    crf.save(tmp_path / "m")
    with pytest.raises(ModelFileError, match="not 'logreg'"):
        TokenLogRegDetector.load(tmp_path / "m")
    path = tmp_path / "m" / "model.crfsuite"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ModelFileError, match="hash"):
        CRFDetector.load(tmp_path / "m")
    with pytest.raises(ModelFileError, match="missing"):
        CRFDetector.load(tmp_path / "nothing")


def test_crf_training_is_deterministic(crf):
    from privgate.ml.crf import CRFDetector

    again = CRFDetector.train(synthetic_docs(), c1=0.01, c2=0.01, max_iterations=60, seed=0)
    assert again.transition_weights() == crf.transition_weights()
