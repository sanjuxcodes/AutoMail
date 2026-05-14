import json
import os
import pandas as pd
import hashlib
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from classifier import classify_grievance
from followup import detect_followup
from priority_llm import get_llm_priority, apply_thread_escalation

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_EMAILS_PATH = os.path.join(BASE_DIR, "demo_emails_from_chunks.json")
EXCEL_FILE = os.path.join(BASE_DIR, "data", "grievances.xlsx")
COLLEGES_PATH = os.path.join(BASE_DIR, "data", "colleges.json")

def generate_id(text):
    return hashlib.md5(text.encode()).hexdigest()[:12]

def clean_for_excel(text):
    if pd.isna(text) or text is None:
        return ""
    text_str = str(text)
    text_str = text_str.replace('\x00', '')
    text_str = ''.join(char if ord(char) < 0x10000 else '' for char in text_str)
    return text_str[:32000]

def main():
    print("🚀 Starting demo data repopulation with priority scoring...")
    
    if not os.path.exists(DEMO_EMAILS_PATH):
        print(f"❌ Error: {DEMO_EMAILS_PATH} not found.")
        return

    with open(DEMO_EMAILS_PATH, "r", encoding="utf-8") as f:
        demo_emails = json.load(f)

    colleges = []
    if os.path.exists(COLLEGES_PATH):
        with open(COLLEGES_PATH, "r", encoding="utf-8") as f:
            colleges = json.load(f)
    
    print(f"📧 Found {len(demo_emails)} demo emails.")

    df = pd.DataFrame()
    
    # Base date for the first email
    base_date = datetime.now() - timedelta(days=len(demo_emails))

    for i, email in enumerate(demo_emails):
        subject = email.get("subject", "")
        body = email.get("body", "")
        content = f"{subject}\n{body}"
        
        email_id = generate_id(content)
        
        # Incremental dates to show a timeline in the frontend
        email_date = (base_date + timedelta(days=i)).date().isoformat()
        
        # Mock sender and institute
        institute = random.choice(colleges) if colleges else "Unknown Institute"
        sender_name = "Department Officer" if "disciplinary" in subject.lower() else "Grievance Applicant"
        sender_email = "officer@university.sh.gov" if "disciplinary" in subject.lower() else f"student_{i}@gmail.com"
            
        category = classify_grievance(content)
        
        # Mocking values for detect_followup and priority
        mock_email_obj = {
            "email_id": email_id,
            "parent_email_id": None,
            "sender": sender_name,
            "subject": subject,
            "content": content,
            "date": email_date
        }
        
        mail_type, parent_id, count = detect_followup(mock_email_obj, df)
        
def get_strict_priority(text, subject, sender_email):
    text_lower = (subject + " " + text).lower()
    
    # Priority 0: Chief Secretary / Highest Authority
    if any(k in text_lower for k in ["chief secretary", "highest authority", "secretary higher education"]):
        return 0
    
    # Priority 1: Court Orders, RTI, Legal Notices
    if any(k in text_lower for k in ["court order", "rti", "legal notice", "show cause notice", "contempt"]):
        return 1
        
    # Priority 2: FIR, Arrest, Misconduct, Suspension
    if any(k in text_lower for k in ["fir", "arrest", "police", "misconduct", "suspension", "disciplinary"]):
        return 2
        
    # Priority 3: CMO, Ministerial, MLA
    if any(k in text_lower for k in ["cmo", "minister", "mla", "governor", "vikas bhawan"]):
        return 3
        
    # Priority 4: Financial urgency
    if any(k in text_lower for k in ["salary", "fund release", "increment", "pay scale", "arrear"]):
        return 4
        
    # Priority 5: Urgent student issues
    if any(k in text_lower for k in ["exam", "result", "admission", "deadline", "correction"]):
        return 5
        
    # Priority 6: General / Feedback (Low)
    return 6

def main():
    print("Starting demo data repopulation with BATCH LLM scoring...")
    
    if not os.path.exists(DEMO_EMAILS_PATH):
        print(f"❌ Error: {DEMO_EMAILS_PATH} not found.")
        return

    with open(DEMO_EMAILS_PATH, "r", encoding="utf-8") as f:
        demo_emails = json.load(f)

    colleges = []
    if os.path.exists(COLLEGES_PATH):
        with open(COLLEGES_PATH, "r", encoding="utf-8") as f:
            colleges = json.load(f)
    
    print("Found " + str(len(demo_emails)) + " demo emails. Preparing batch...")

    # First pass: Generate IDs and prepare for batch LLM
    batch_data = []
    for i, e in enumerate(demo_emails):
        subject = e.get("subject", "")
        body = e.get("body", "")
        content = f"{subject}\n{body}"
        email_id = generate_id(content)
        
        batch_data.append({
            "id": email_id,
            "subject": subject,
            "sender": "Grievance Applicant",  # Placeholder for prompt
            "content": content
        })

    from priority_llm import get_batch_priorities
    print("Calling Gemini for BATCH prioritisation (this should take ~15s)...")
    batch_results = get_batch_priorities(batch_data)
    print("Received " + str(len(batch_results)) + " scores from LLM.")

    df = pd.DataFrame()
    base_date = datetime.now() - timedelta(days=len(demo_emails))

    for i, email in enumerate(demo_emails):
        subject = email.get("subject", "")
        body = email.get("body", "")
        content = f"{subject}\n{body}"
        email_id = generate_id(content)
        
        email_date = (base_date + timedelta(days=i)).date().isoformat()
        institute = random.choice(colleges) if colleges else "Unknown Institute"
        sender_name = "Department Officer" if any(k in subject.lower() for k in ["disciplinary", "suspension"]) else "Grievance Applicant"
        sender_email = "officer@university.sh.gov" if any(k in subject.lower() for k in ["disciplinary", "suspension"]) else f"student_{i}@gmail.com"
            
        category = classify_grievance(content)
        
        # Priority Logic: Batch Result -> Heuristic Fallback
        final_p = batch_results.get(email_id)
        if final_p is None:
            # Fallback to keyword-based heuristic
            final_p = get_strict_priority(content, subject, sender_email)
            print(f"   - Fallback (Heuristic) for: {subject[:30]}")
        
        # Follow-up detection and thread escalation
        mail_type, parent_id, count = detect_followup({
            "email_id": email_id,
            "parent_email_id": None,
            "sender": sender_name,
            "subject": subject,
            "content": content,
            "date": email_date
        }, df)
        
        final_p = apply_thread_escalation(final_p, count)
        
        row = {
            "email_id": email_id,
            "parent_email_id": None,
            "sender": sender_name,
            "sender_email": sender_email,
            "institute_name": institute,
            "subject": subject,
            "date": email_date,
            "content": content,
            "attachments": "[]",
            "eml_file": f"demo_{i}.eml",
            "category": category,
            "mail_type": mail_type,
            "followup_count": count,
            "priority": final_p,
            "is_demo": True
        }
        
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Clean text fields
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(clean_for_excel)

    # Save results
    os.makedirs(os.path.dirname(EXCEL_FILE), exist_ok=True)
    df.to_excel(EXCEL_FILE, index=False)
    print("Successfully saved " + str(len(df)) + " records with AI BATCH heuristic scores to " + EXCEL_FILE)

if __name__ == "__main__":
    main()
