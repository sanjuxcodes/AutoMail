import re
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def has_explicit_reference(subject, content):
    """Check if email has explicit references to previous communication"""
    text = (subject + " " + content).lower()
    
    # Check for common follow-up indicators
    followup_patterns = [
        r'\bre:?\s',  # Re: or Re
        r'\bfwd:?\s',  # Fwd: or Fwd
        r'\bregarding previous',
        r'\bas mentioned',
        r'\bas per previous',
        r'\bin continuation',
        r'\bfollowing up',
        r'\bfollow-up',
        r'\bearlier mail',
        r'\bprevious mail',
        r'\bprevious email',
        r'\blast mail',
        r'\bagain sending',
        r'\bresending',
        r'\bresubmit',
    ]
    
    for pattern in followup_patterns:
        if re.search(pattern, text):
            return True
    return False

def detect_followup(email, df):
    # A. Threading (best) - Check if email explicitly references a parent email
    if email["parent_email_id"]:
        if email["parent_email_id"] in df["email_id"].values:
            parent = df[df["email_id"] == email["parent_email_id"]].iloc[0]
            return "Follow-up", parent["email_id"], parent["followup_count"] + 1

    # B. Check emails from the SAME sender (college/person)
    if not df.empty and email.get("sender"):
        # Get all previous emails from the same sender
        same_sender_df = df[df["sender"] == email["sender"]]
        
        # If this is the first email from this sender → Fresh
        if same_sender_df.empty:
            return "Fresh", None, 0
        
        # Check for explicit references (Re:, Fwd:, "regarding previous", etc.)
        subject = email.get("subject", "")
        content = email.get("content", "")
        
        if has_explicit_reference(subject, content):
            # Find the most recent email from same sender as parent
            parent = same_sender_df.iloc[-1]
            return "Follow-up", parent["email_id"], parent["followup_count"] + 1
        
        # C. Check semantic similarity with previous emails from SAME sender
        # Only if they have similar content, mark as follow-up
        recent_df = same_sender_df.tail(10)  # Check last 10 emails from same sender
        old_texts = recent_df["content"].tolist()
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            model = get_model()
            emb_old = model.encode(old_texts, show_progress_bar=False)
            emb_new = model.encode([content], show_progress_bar=False)
            sims = cosine_similarity(emb_new, emb_old)[0]

            # High similarity threshold - only mark as follow-up if very similar
            if sims.max() > 0.88:
                parent_idx = sims.argmax()
                parent = recent_df.iloc[parent_idx]
                return "Follow-up", parent["email_id"], parent["followup_count"] + 1
        except Exception as e:
            print(f"⚠️ Similarity check failed: {e}")

    # Default: Fresh email (first time from sender OR different topic)
    return "Fresh", None, 0
