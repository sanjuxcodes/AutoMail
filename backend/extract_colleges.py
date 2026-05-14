import pandas as pd
import re
import json

df = pd.read_excel('data/grievances.xlsx')

# Extract all unique college names from senders
unique_colleges = set()
for sender in df['sender'].dropna():
    match = re.match(r'^(.+?)\s*<', str(sender))
    if match:
        college_name = match.group(1).strip().strip('"').strip("'")
        if len(college_name) > 3:
            unique_colleges.add(college_name)

# Load existing colleges
with open('data/colleges.json', 'r', encoding='utf-8') as f:
    colleges = json.load(f)

# Add new ones
existing_set = set(c.lower() for c in colleges)
new_count = 0
for college in sorted(unique_colleges):
    if college.lower() not in existing_set:
        colleges.append(college)
        new_count += 1

# Save
with open('data/colleges.json', 'w', encoding='utf-8') as f:
    json.dump(colleges, f, indent=2, ensure_ascii=False)

print(f'Added {new_count} new colleges from email senders. Total: {len(colleges)}')
