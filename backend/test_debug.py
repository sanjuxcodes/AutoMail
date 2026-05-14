import os, sys, json
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from rag.reply_generator import generate_suggested_reply_with_evidence

print("--- Calling generator ---")
res = generate_suggested_reply_with_evidence(
    email_content="I applied 30 days ago and heard nothing.",
    email_subject="Scholarship delay",
    sender="test sender"
)

print("\n--- Result ---")
print(json.dumps(res, indent=2))
