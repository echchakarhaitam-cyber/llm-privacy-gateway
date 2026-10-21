"""Feature extraction: documented keys, determinism, stacked regex feature."""

from privgate.ml.features import FEATURE_VERSION, doc_features, word_features
from privgate.ml.tokens import tokenize


def test_features_are_deterministic_and_documented():
    text = "Dr. Smith wrote to a@b.org on 2024-01-05."
    tokens = tokenize(text)
    first = doc_features(text, tokens)
    assert first == doc_features(text, tokens) and len(first) == len(tokens)
    smith = first[2]
    assert smith["0:lower"] == "smith" and smith["0:shape"] == "Xxxxx"
    assert smith["0:sshape"] == "Xx" and smith["0:title"] == 1.0
    assert smith["-1:lower"] == "." and smith["-2:lower"] == "dr" and smith["+1:lower"] == "wrote"
    assert first[0]["BOS"] == 1.0 and first[-1]["EOS"] == 1.0
    at = next(f for f in first if f["0:lower"] == "@")
    assert at["0:hasat"] == 1.0 and at["0:punct"] == 1.0 and at["regex"] == "EMAIL"
    assert first[2]["regex"] == "O"
    assert dict(word_features("2024"))["digit"] == 1.0
    assert "upper" not in dict(word_features("abc"))   # false flags are absent
    assert FEATURE_VERSION
