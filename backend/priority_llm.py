import os
import re
from google import genai
from dotenv import load_dotenv

load_dotenv()

_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=_api_key) if _api_key else None

def get_llm_priority(email_text, subject="", sender=""):
    prompt = f"""
You are an expert email triage assistant for a government education department.

Analyze this email carefully considering:
- The SENDER's authority level
- The URGENCY and time-sensitivity of the issue
- The LEGAL or administrative implications
- The IMPACT on students or institutions

Then assign ONE priority rank from below:

0: Chief Secretary / highest authority communication (Critical)
1: Court orders, RTI, legal notices (Immediate Action - Critical)
2: Legal disputes, FIR cases, serious misconduct, suspension cases (High)
3: CMO office, ministerial, MLA-level, or higher education office matters (High)
4: Financial urgency: Fund release, salary disbursement, increment issues (Medium)
5: Urgent student issues: Exams, results, admission deadlines (Medium)
6: General grievances, routine queries, portal issues, or feedback (Low Priority)

Rules:
- If the email expresses frustration, repeated follow-ups, or mentions deadlines, increase urgency by 1 level (lower number).
- If the sender holds a high official position, prioritize accordingly.
- Return ONLY a single digit (0-6). No explanation.

Email:
Sender: {sender}
Subject: {subject}
Content: {email_text}
"""

    try:
        if client is None:
            return 5
        
        # Try Gemini 2.0 Flash first
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
        except Exception:
            # Fallback to Gemini Flash Latest
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
            )
            
        result = (getattr(response, "text", "") or "").strip()

        match = re.search(r"\d+", result)
        if match:
            return int(match.group())

        return 5
    except Exception:
        return 5

import json

def get_batch_priorities(emails):
    """
    Emails: list of {id, subject, sender, content}
    Returns: dict of {id: priority}
    """
    if not emails or client is None:
        return {}

    # Format emails into a single block
    emails_block = ""
    for e in emails:
        emails_block += f"ID: {e['id']}\nSENDER: {e['sender']}\nSUBJECT: {e['subject']}\nCONTENT: {e['content'][:500]}...\n---\n"

    prompt = f"""
You are an expert triage assistant for a government education department. 
Analyze the following batch of emails and assign a priority rank (0-6) to each.

PRIORITY LEVELS (0-6):
0: Chief Secretary / highest authority (Critical)
1: Court orders, RTI, legal notices (Immediate Action - Critical)
2: Legal disputes, FIR cases, misconduct, suspension (High)
3: CMO office, ministerial, MLA, higher education office (High)
4: Financial urgency: Salary, fund release, increments (Medium)
5: Urgent student issues: Exams, results, admission deadlines (Medium)
6: General grievances, routine queries, feedback (Low Priority)

Rules:
- If follow-up mentions or deadlines exist, increase urgency (lower number).
- Return YOUR RESPONSE AS A VALID JSON OBJECT ONLY.
- Format: {{ "EMAIL_ID": PRIORITY_NUMBER, ... }}

EMAILS TO PROCESS:
{emails_block}
"""

    try:
        # Try Gemini 2.0 Flash first
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={ 'response_mime_type': 'application/json' }
            )
        except Exception:
            # Fallback to Gemini Flash Latest
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
                config={ 'response_mime_type': 'application/json' }
            )
        
        result_text = getattr(response, "text", "{}")
        priorities = json.loads(result_text)
        
        # Ensure all IDs are strings and values are ints
        return {str(k): int(v) for k, v in priorities.items()}
    except Exception as e:
        print("Batch LLM scoring failed: " + str(e))
        return {}

def apply_thread_escalation(priority, followup_count):
    # If many follow-ups, increase priority
    if followup_count > 3 and priority in [4, 5]:
        return 2  # escalate to legal-level seriousness
    return priority