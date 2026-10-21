"""Dataset statistics, and the generated blocks of docs/DATA.md staying in sync with code."""

import re
from pathlib import Path

from privgate.data.documents import Document
from privgate.data.mapping import LABEL_MAPS, mapping_table_markdown
from privgate.data.stats import SetStats, covered_characters
from privgate.spans import EntityType, Span

P, LOC = EntityType.PERSON, EntityType.LOCATION
DATA_MD = Path(__file__).resolve().parents[1] / "docs" / "DATA.md"


def test_covered_characters_counts_overlaps_once():
    doc = Document("d", "x" * 20, (Span(0, 10, P), Span(5, 12, LOC), Span(15, 18, P)), "s")
    assert covered_characters(doc) == 15


def test_set_stats():
    stats = SetStats()
    stats.add(Document("a", "John here", (Span(0, 4, P),), "s", {"domain": "Health"}))
    stats.add(Document("b", "nothing", (), "s", {"domain": "Health"}))
    stats.add(Document("c", "x", (), "s"))
    d = stats.to_dict()
    assert (d["documents"], d["characters"], d["pii_characters"], d["spans"]) == (3, 17, 4, 1)
    assert d["pii_char_share"] == round(4 / 17, 6)
    assert d["spans_per_label"]["PERSON"] == 1 and d["spans_per_label"]["EMAIL"] == 0
    assert d["top_domains"] == [["Health", 2], ["(none)", 1]]
    assert SetStats().pii_char_share == 0.0


def test_mapping_table_in_docs_is_generated_from_code():
    docs = DATA_MD.read_text(encoding="utf-8")
    block = re.search(r"<!-- mapping:begin -->\n(.*?)\n<!-- mapping:end -->", docs, re.DOTALL)
    assert block, "docs/DATA.md lacks the generated mapping block"
    assert block.group(1).strip() == mapping_table_markdown().strip()
    for table in LABEL_MAPS.values():
        for label in table:
            assert f"`{label}`" in block.group(1)
