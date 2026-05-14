import os
import json
import hashlib
from typing import Dict, Any, List
from dotenv import load_dotenv

from rag.retriever import retrieve_policy_chunks

from google import genai

# ---------- CONFIG ----------
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY or "your_google_api_key" in API_KEY or "your_gemini_api_key" in API_KEY:
    print(f"[ERROR]: Gemini API key is missing. KEY_LEN: {len(API_KEY) if API_KEY else 0}")

client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print(f"[ERROR]: Gemini Client Init Failed: {e}")

# ---------- SYSTEM PROMPT ----------
SYSTEM_PROMPT = """
You are a senior professional grievance redressal officer for an educational institution.

Primary Goal: Generate a formal, helpful, and policy-grounded email reply.

Rules:
1. Strict Grounding: Use ONLY the provided policy context.
2. Complete Response: Output a FULL email (Subject, Salutation, Body, Signature).
3. Minimum Length: The body must be 3-5 sentences minimum.
4. Policy Citation: You MUST explicitly mention specific clause IDs (e.g., "As per MSC-06...") to support your statements.
5. Delay Handling: If the user query is about a delay or pending status, you MUST mention the specific escalation path (e.g., nodal officer) or response timelines from the policy.
6. Tone: Professional, authoritative yet respectful.
7. Language: Formal English.

If the provided policy context is insufficient, weak, or unrelated:
- DO NOT attempt to infer or combine unrelated clauses
- Respond stating that no relevant policy clause is available
- Do not fabricate policy references
"""

FALLBACK_TEMPLATE = """
Subject: RE: {subject}

Dear Sir/Madam,

Your grievance has been received and is under review. 
The institution will process the matter as per applicable policy provisions.

Sincerely,
Grievance Support Desk
""".strip()


def build_prompt(subject: str, sender: str, email: str, chunks: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"[Clause {c['metadata']['clause_id']}] {c['text']}"
        for c in chunks
    )

    return f"""
{SYSTEM_PROMPT}

USER EMAIL:
Subject: {subject}
From: {sender}
Content: {email}

POLICY CONTEXT:
{context}

TASK:
Write a professional reply email to the student/staff member. 
- You MUST use the policy context provided.
- You MUST mention specific clause IDs (e.g., MSC-03, SUS-01).
- If there's a delay mentioned (e.g., 30 days pending), refer back to the specific delay/escalation rules in the policy chunks.
- Ensure the email has a clear Subject (RE: {subject}), starts with "Dear {sender}," and ends with "Sincerely, Grievance Support Desk".

OUTPUT FORMAT:
Subject: RE: {subject}
Dear {sender},

[Professional body using policy context and referencing clause IDs]

Sincerely,
Grievance Support Desk
""".strip()


def generate_reply(subject: str, sender: str, email: str, chunks: List[Dict[str, Any]]) -> str:
    if not client:
        return ""

    prompt = build_prompt(subject, sender, email, chunks)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
    except Exception as e_primary:
        print(f"[WARN]: gemini-2.0-flash failed ({e_primary}), falling back to gemini-flash-latest...")
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
            )
        except Exception as e_secondary:
            print(f"[ERROR]: Both LLM Generations Error: {e_secondary}")
            return ""

    reply = ""
    if response and response.text:
        reply = response.text.strip()
        
    # Task 2: Safety & Length checks
    if not reply or len(reply) < 30:
        print(f"[WARN]: LLM response too short ({len(reply) if reply else 0} chars)")
        return ""
        
    return reply


def generate_suggested_reply_with_evidence(
    email_content: str,
    email_subject: str = "",
    sender: str = "",
    category: str = ""
) -> Dict[str, Any]:
    """
    Main entry point for suggested reply generation.
    Ensures safe JSON output and provides mandatory debug logs.
    """
    
    # Task 5 & 6: Fix Sender + Subject + "nan" bug
    clean_sender = str(sender) if sender and str(sender).lower() != "nan" else "Sir/Madam"
    clean_subject = str(email_subject) if email_subject and str(email_subject).lower() != "nan" else "Your Inquiry"

    query = f"{clean_subject} {email_content}".strip()
    
    # Retrieval
    retrieved = []
    try:
        retrieved = retrieve_policy_chunks(query)
    except Exception as e:
        print(f"[ERROR]: Retrieval failed: {e}")
        retrieved = []

    # Task 3: SAFE FALLBACK for empty Context
    if not retrieved:
        status = "no_policy"
        print("[WARN]: No relevant policy chunks found. Applying safe fallback.")
        reply = f"""Subject: RE: {clean_subject}
Dear {clean_sender},
We regret to inform you that no specific institutional policy clause directly addresses your query.
Your concern has been forwarded to the concerned authority for review.
Sincerely,
Grievance Support Desk"""
        
        evidence = []
    else:
        # Generation
        reply = ""
        status = "ok"
        reply = generate_reply(clean_subject, clean_sender, email_content, retrieved)
        
        # Smarter Fallback: lenient, case-insensitive check
        # Accept any common professional closing — don't throw away valid replies
        VALID_CLOSINGS = ("sincerely", "regards", "yours faithfully", "best regards", "with regards", "yours truly")
        reply_lower = reply.lower() if reply else ""
        has_greeting = "dear" in reply_lower
        has_closing = any(c in reply_lower for c in VALID_CLOSINGS)
    
        if not reply or len(reply) < 80 or not has_greeting or not has_closing:
            print(f"[WARN]: Fallback triggered -> has_greeting={has_greeting}, has_closing={has_closing}, reply_len={len(reply) if reply else 0}")
            reply = FALLBACK_TEMPLATE.format(subject=clean_subject)
            status = "fallback"

        # Formatting evidence
        evidence = [
            {
                "clause_id": c["metadata"].get("clause_id", "N/A"),
                "section": c["metadata"].get("section", "General"),
                "score": round(c.get("score", 0.0), 3),
                "text": str(c.get("text", "")).strip()
            }
            for c in retrieved
        ]

    # Task 8: MANDATORY Debug Logs
    top_score = evidence[0]["score"] if evidence else 0.0
    clause_ids = [e["clause_id"] for e in evidence]
    resp_len = len(reply)
    
    print("-" * 50)
    print(f"[DEBUG] Top retrieval score: {top_score}")
    print(f"[DEBUG] Retrieved clause IDs: {clause_ids}")
    print(f"[DEBUG] LLM response length: {resp_len}")
    print(f"[DEBUG] Final status: {status}")
    print("-" * 50)

    # Task 7: NEVER break frontend
    return {
        "suggested_reply": reply,
        "evidence": evidence,
        "status": status
    }


def generate_suggested_reply(email_content: str, email_subject: str = "", sender: str = "") -> str:
    result = generate_suggested_reply_with_evidence(email_content, email_subject, sender)
    return result.get("suggested_reply", "Error generating reply")
