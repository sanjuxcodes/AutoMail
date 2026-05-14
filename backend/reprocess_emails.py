import os
import pandas as pd
from email_parser import parse_eml
from classifier import classify_grievance
from followup import detect_followup

# Load existing data
df = pd.read_excel('data/grievances.xlsx')
print(f'Original: {len(df[df["mail_type"] == "Fresh"])} Fresh, {len(df[df["mail_type"] == "Follow-up"])} Follow-up')

# Reset all to Fresh and reprocess
df['mail_type'] = 'Fresh'
df['parent_email_id'] = None
df['followup_count'] = 0

# Reprocess each email with new logic
processed_df = pd.DataFrame()
for idx, row in df.iterrows():
    email_dict = row.to_dict()
    mail_type, parent_id, followup_count = detect_followup(email_dict, processed_df)
    email_dict['mail_type'] = mail_type
    email_dict['parent_email_id'] = parent_id
    email_dict['followup_count'] = followup_count
    processed_df = pd.concat([processed_df, pd.DataFrame([email_dict])], ignore_index=True)
    if idx % 50 == 0:
        print(f'Processed {idx}/{len(df)} emails...')

# Save updated data
processed_df.to_excel('data/grievances.xlsx', index=False)
print(f'\nUpdated: {len(processed_df[processed_df["mail_type"] == "Fresh"])} Fresh, {len(processed_df[processed_df["mail_type"] == "Follow-up"])} Follow-up')
print('Excel file updated successfully!')
