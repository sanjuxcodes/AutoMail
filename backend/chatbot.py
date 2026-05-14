import pandas as pd
import re
from datetime import timedelta

EXCEL_FILE = "./data/grievances.xlsx"

# ---------------- TYPO FIXING ----------------
TYPO_MAP = {
    'hoe': 'how',
    'meny': 'many',
    'mny': 'many',
    'emials': 'emails',
    'emals': 'emails',
    'frm': 'from',
    'suspention': 'suspension',
    'suspnsion': 'suspension',
    'followup': 'follow up',
    'folowup': 'follow up',
    'collge': 'college',
    'colege': 'college',
}

def fix_typos(text: str) -> str:
    words = text.lower().split()
    return " ".join(TYPO_MAP.get(w, w) for w in words)

# ---------------- QUESTION PARSER ----------------
def parse_question(question: str) -> dict:
    question = fix_typos(question)
    q = question.lower()

    parsed = {
        "action": "count",
        "category": None,
        "sender": None,
        "days": None,
        "mail_type": None,
        "attachments": None
    }

    # Attachment intent
    if "attachment" in q or "attachments" in q:
        if any(w in q for w in ["no", "not", "without"]):
            parsed["attachments"] = False
        else:
            parsed["attachments"] = True

    # Action
    if any(w in q for w in ["show", "list", "display"]):
        parsed["action"] = "list"

    # -------- CATEGORY NORMALIZATION --------
    CATEGORY_MAP = {
        "suspension": ["suspension", "disciplinary"],
        "fir": ["fir", "arrest", "police"],
        "semester": ["semester", "exam"]
    }

    for cat, keys in CATEGORY_MAP.items():
        if any(k in q for k in keys):
            parsed["category"] = cat
            break

    # -------- FOLLOW-UP / FRESH --------
    if any(k in q for k in ["follow up", "follow-up", "followups", "follow ups"]):
        parsed["mail_type"] = "Follow-up"
    elif "fresh" in q:
        parsed["mail_type"] = "Fresh"

    # -------- COLLEGE EXTRACTION --------
    college_match = re.search(
        r'from\s+([a-zA-Z\s\.\-&]+?)(?:\s+college|\s+university|\s+institute|$)',
        question,
        re.IGNORECASE
    )
    if college_match:
        parsed["sender"] = college_match.group(1).strip().lower()

    # -------- DATE PARSING --------
    # last X days
    days_match = re.search(r'last\s+(\d+)\s*days?', q)
    if days_match:
        parsed["days"] = int(days_match.group(1))

    # last X months
    months_match = re.search(r'last\s+(\d+)\s*months?', q)
    if months_match:
        parsed["days"] = int(months_match.group(1)) * 30

    print("DEBUG QUESTION:", question)
    print("DEBUG PARSED:", parsed)

    return parsed

# ---------------- QUERY EXECUTION ----------------
def run_query(parsed: dict) -> str:
    df = pd.read_excel(EXCEL_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["has_attachments"] = df["attachments"].apply(
    lambda x: bool(str(x).strip()) and str(x).strip() not in ["[]", "nan", ""])

    df["sender_clean"] = (
        df["sender"]
        .str.lower()
        .str.replace(r'[^a-z\s]', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    original_count = len(df)

    # -------- APPLY FILTERS --------
    if parsed["category"]:
        df = df[df["category"].str.contains(parsed["category"], case=False, na=False)]

    sender_key = (
    parsed["sender"]
    .lower()
    .replace("government", "")
    .replace("govt", "")
    .replace("college", "")
    .strip()
    )

    df = df[df["sender_clean"].str.contains(sender_key, na=False)]

    if parsed["attachments"] is True:
        df = df[df["has_attachments"] == True]

    elif parsed["attachments"] is False:
        df = df[df["has_attachments"] == False]

    if parsed["mail_type"]:
        df = df[df["mail_type"] == parsed["mail_type"]]

    if parsed["days"]:
        max_date = df["date"].max()
        cutoff = max_date - timedelta(days=parsed["days"])
        df = df[df["date"] >= cutoff]

    # -------- FINAL SAFETY CHECK --------
    if parsed != {
        "action": "count",
        "category": None,
        "sender": None,
        "days": None,
        "mail_type": None,
        "attachments": None 
    } and len(df) == original_count:
        print("⚠️ WARNING: Filters detected but no reduction happened")

    # -------- COUNT RESPONSE --------
    if parsed["action"] == "count":
        parts = []

        if parsed["category"]:
            parts.append(f"category '{parsed['category']}'")
        if parsed["mail_type"]:
            parts.append(parsed["mail_type"].lower())
        if parsed["sender"]:
            parts.append(f"from '{parsed['sender']}'")
        if parsed["days"]:
            parts.append(f"in last {parsed['days']} days")

        desc = " ".join(parts) if parts else "in the database"

        return f"{len(df)} email{'s' if len(df) != 1 else ''} {desc}."

    # -------- LIST RESPONSE --------
    if df.empty:
        return "No matching records found."

    return "\n".join(
        f"{i+1}. {row['subject']} ({row['date'].date()})"
        for i, row in df.head(10).iterrows()
    )

# ---------------- ENTRY POINT ----------------
def ask(question: str) -> str:
    parsed = parse_question(question)
    return run_query(parsed)
