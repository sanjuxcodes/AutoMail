import sys
import os
import json

# Add current directory to path so we can import rag
sys.path.append(os.getcwd())

from rag.reply_generator import generate_suggested_reply_with_evidence

def test_query(subject, content, sender="John Doe"):
    print(f"\n--- Testing Query: {subject} ---")
    result = generate_suggested_reply_with_evidence(content, subject, sender)
    
    print(f"Status: {result['status']}")
    print(f"Suggested Reply (first 100 chars): {result['suggested_reply'][:100]}...")
    print(f"Evidence count: {len(result['evidence'])}")
    for i, ev in enumerate(result['evidence'][:2]):
        print(f"  [{i+1}] Clause: {ev['clause_id']}, Section: {ev['section']}, Score: {ev['score']}")

if __name__ == "__main__":
    # Test cases
    test_query(
        "Scholarship application not processed within 30 days",
        "I applied for a scholarship 40 days ago but still haven't heard back. Please check.",
        sender="nan"
    )
    
    test_query(
        "Exam results delayed",
        "My semester exam results are not yet published.",
        sender="Student"
    )

    test_query(
        "Hostel safety concern",
        "The hostel lock is broken and I feel unsafe.",
        sender=""
    )
