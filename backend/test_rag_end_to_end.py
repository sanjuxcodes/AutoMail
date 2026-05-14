import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_generate_reply(subject, content):
    print(f"\n--- Testing: {subject} ---")
    payload = {
        "email_subject": subject,
        "email_content": content,
        "sender": "Test User",
        "category": "miscellaneous"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/generate-reply", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result.get('status')}")
            print(f"Reply Preview: {result.get('suggested_reply')[:200]}...")
            print(f"Evidence Found: {len(result.get('evidence', []))}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    # Just ONE test at a time to avoid 429 on free tier
    test_generate_reply("Scholarship delay", "My scholarship payment has been delayed for 3 months. What is the procedure to resolve this?")
