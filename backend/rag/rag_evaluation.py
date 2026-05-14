import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from rag.reply_generator import generate_suggested_reply_with_evidence


@dataclass
class EvaluationCase:
    name: str
    email_content: str
    email_subject: str = ""
    sender: str = ""


TEST_CASES: List[EvaluationCase] = [
    EvaluationCase(
        name="Time limit grievance",
        email_subject="Delayed complaint submission",
        sender="principal@example.edu",
        email_content=(
            "We are submitting a grievance regarding scholarship delay that happened 4 months ago. "
            "Please advise if this can still be accepted and what details are required."
        ),
    ),
    EvaluationCase(
        name="Examination correction",
        email_subject="Result correction request",
        sender="student@example.edu",
        email_content=(
            "I need rechecking and correction of my semester examination result. "
            "Please guide me on process and expected communication."
        ),
    ),
    EvaluationCase(
        name="Out of scope random request",
        email_subject="Hostel food menu complaint",
        sender="user@example.edu",
        email_content=(
            "Please redesign the weekly hostel food menu and provide a nutrition chart "
            "for all hostels immediately."
        ),
    ),
]


def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = result.get("retrieval", {})
    grounding = result.get("grounding_report", {})

    return {
        "generation_mode": result.get("generation_mode"),
        "retrieved_count": len(retrieval.get("matches", [])),
        "is_grounded": grounding.get("is_grounded"),
        "supported_sentence_ratio": grounding.get("supported_sentence_ratio"),
        "top_clauses": [
            m.get("metadata", {}).get("clause_id")
            for m in retrieval.get("matches", [])[:3]
        ],
    }


def main() -> None:
    report: List[Dict[str, Any]] = []

    for case in TEST_CASES:
        result = generate_suggested_reply_with_evidence(
            email_content=case.email_content,
            email_subject=case.email_subject,
            sender=case.sender,
        )

        entry = {
            "case": case.name,
            "input": {
                "subject": case.email_subject,
                "sender": case.sender,
                "email_content": case.email_content,
            },
            "summary": summarize(result),
            "result": result,
        }
        report.append(entry)

    output_path = Path(__file__).resolve().parent / "data" / "rag_evaluation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("RAG evaluation complete")
    print(f"Report saved to: {output_path}")
    for item in report:
        s = item["summary"]
        print(
            f"- {item['case']}: mode={s['generation_mode']} "
            f"retrieved={s['retrieved_count']} grounded={s['is_grounded']} "
            f"ratio={s['supported_sentence_ratio']}"
        )


if __name__ == "__main__":
    main()
