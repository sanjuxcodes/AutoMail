import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rag_pipeline.ingest import CHUNKS_PATH, INDEX_PATH, ingest_policy_document

DEMO_DIR = BASE_DIR / "demo_policies"
DATA_DIR = BASE_DIR / "rag" / "data"


def main() -> None:
    pdf_candidates = list(DEMO_DIR.glob("*.pdf")) + list(DEMO_DIR.glob("*.PDF"))
    pdfs = sorted({p.resolve() for p in pdf_candidates})
    print(f"pdf_count {len(pdfs)}")

    CHUNKS_PATH.unlink(missing_ok=True)
    INDEX_PATH.unlink(missing_ok=True)

    ingested = 0
    for p in pdfs:
        try:
            ingest_policy_document(str(p), policy_id=p.stem, section_title="Demo Policy")
            ingested += 1
        except Exception as exc:
            print(f"ingest_fail {p.name} {exc}")

    print(f"ingested {ingested}")

    exp = [
        " ".join(str(c.get("text", "")).split())
        for c in json.load(open(DATA_DIR / "demo_policy_chunks.json", encoding="utf-8"))
    ]
    act = [
        " ".join(str(c.get("text", "")).split())
        for c in json.load(open(DATA_DIR / "policy_chunks.json", encoding="utf-8"))
    ]

    print(f"expected {len(exp)}")
    print(f"actual {len(act)}")
    print(f"exact_match {exp == act}")

    if exp != act:
        exp_set = set(exp)
        act_set = set(act)
        missing = [x for x in exp if x not in act_set]
        extra = [x for x in act if x not in exp_set]
        print(f"missing_count {len(missing)}")
        print(f"extra_count {len(extra)}")
        if missing:
            print(f"sample_missing {missing[0]}")
        if extra:
            print(f"sample_extra {extra[0]}")


if __name__ == "__main__":
    main()
