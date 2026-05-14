import os
import pandas as pd
from email_parser import parse_eml
from classifier import classify_grievance
from followup import detect_followup

FILES_DIR = "./files"
EXCEL_FILE = "./data/grievances.xlsx"

def clean_for_excel(text):
    if pd.isna(text) or text is None:
        return ""
    text_str = str(text)
    text_str = text_str.replace('\x00', '')
    text_str = ''.join(char if ord(char) < 0x10000 else '' for char in text_str)
    return text_str[:32000]

print("🚀 Starting email processing...")
print(f"📂 Scanning directory: {FILES_DIR}")

eml_files = sorted([f for f in os.listdir(FILES_DIR) if f.lower().endswith(".eml")])
print(f"📧 Found {len(eml_files)} .eml files")

df = pd.DataFrame()
processed_count = 0

for i, filename in enumerate(eml_files, 1):
    file_path = os.path.join(FILES_DIR, filename)
    print(f"\n[{i}/{len(eml_files)}] Processing: {filename}")
    
    email = parse_eml(file_path)
    if not email:
        print(f"⚠️ Skipped (parsing failed)")
        continue
    
    # Check if already processed
    if not df.empty and email["email_id"] in df["email_id"].values:
        print(f"✓ Already processed")
        continue
    
    category = classify_grievance(email["content"])
    mail_type, parent_id, count = detect_followup(email, df)
    
    row = {
        **email,
        "category": category,
        "mail_type": mail_type,
        "followup_count": count,
    }
    
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    processed_count += 1
    print(f"✓ Added to database - Date: {email.get('date', 'N/A')} - Category: {category} - Type: {mail_type}")
    
    if i % 50 == 0:
        print(f"\n📊 Progress: {i}/{len(eml_files)} files processed")

# Clean all text fields for Excel compatibility
print("\n🧹 Cleaning data for Excel compatibility...")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(clean_for_excel)

# Ensure data directory exists
os.makedirs(os.path.dirname(EXCEL_FILE), exist_ok=True)

df.to_excel(EXCEL_FILE, index=False)
print(f"\n✅ Processing complete! Total records: {len(df)}")
print(f"📊 Data saved to: {EXCEL_FILE}")
print(f"\n📅 Date statistics:")
print(f"   Unique dates: {df['date'].nunique()}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\n📧 Mail type distribution:")
print(df['mail_type'].value_counts())
