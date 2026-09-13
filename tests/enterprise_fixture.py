"""Small synthetic fixture without network, checkpoints or optional ML imports."""
from collections import Counter
from datetime import timedelta

from enterprise_dataset import EnterpriseDataset
from enterprise_seed import START, build_dataset, iso


def fixture(requests=80):
    raw = build_dataset(requests=requests)
    manifest = {
        "dataset": "enterprise_knowledge_recommendation_synthetic_pilot",
        "version": "1.0.0", "seed": 20260913, "synthetic": True, "request_count": requests,
        "time_range": {"start": iso(START), "end": iso(START + timedelta(days=90))},
        "split_counts": dict(Counter(r["split"] for r in raw["recommendations.jsonl"])),
        "event_counts": dict(Counter(r["event_type"] for r in raw["events.jsonl"])),
    }
    return raw, manifest, EnterpriseDataset.from_rows(raw, manifest)
