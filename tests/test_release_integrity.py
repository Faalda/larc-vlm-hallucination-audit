import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_population():
    records = json.loads(
        (ROOT / "results" / "pilot_records_caption_corrected.json").read_text()
    )
    assert len(records) == 1649
    assert len({record["image_id"] for record in records}) == 592
    assert sum(not record["grounded_corrected"] for record in records) == 126


def test_public_audit():
    path = ROOT / "results" / "e4_label_audit" / "audit_labels.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == 200
    counts = {
        label: sum(row["final_label"] == label for row in rows)
        for label in ("VISIBLE", "NOT_VISIBLE", "AMBIGUOUS")
    }
    assert counts == {"VISIBLE": 134, "NOT_VISIBLE": 61, "AMBIGUOUS": 5}
    forbidden = {"caption", "gtc_any_object", "align_cos", "updated_at"}
    assert all(not forbidden.intersection(row) for row in rows)


def test_indexes_use_relative_paths():
    for path in (ROOT / "data").glob("*_index*.json"):
        for row in json.loads(path.read_text()):
            local_path = Path(row["local_path"])
            assert not local_path.is_absolute()
            assert local_path.parts[0] == "data"


def test_manuscript_is_not_released():
    assert not (ROOT / "paper").exists()
