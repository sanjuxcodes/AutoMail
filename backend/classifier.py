import re
from collections import defaultdict

# Expanded keyword lists with variations and synonyms
CATEGORIES = {
    "Suspension / Disciplinary": [
        # Core terms
        "suspension", "suspended", "suspend",
        # Disciplinary terms
        "chargesheet", "charge sheet", "charge-sheet",
        "disciplinary", "discipline", "disciplinary action",
        "departmental proceeding", "departmental proceedings",
        "departmental inquiry", "departmental enquiry",
        # Related terms
        "dismissal", "dismissed", "termination",
        "punishment", "penalty", "penal action",
        "misconduct", "violation", "breach",
        "show cause", "show-cause", "showcause",
        "inquiry", "enquiry", "investigation",
        "disciplinary case", "disciplinary matter",
        # Context-specific
        "suspension case", "suspension cases",
        "suspended employee", "suspended staff",
        "suspended official", "suspended officer",
        "suspension report", "suspension details",
        "suspension information", "suspension status"
    ],
    "FIR / Arrest / Legal": [
        # Legal terms
        "fir", "first information report",
        "arrest", "arrested", "arresting",
        "police", "police case", "police complaint",
        "court", "court case", "court order",
        "legal", "legal case", "legal matter",
        "litigation", "lawsuit", "suit",
        # Criminal terms
        "criminal", "criminal case", "criminal complaint",
        "complaint", "complaint filed",
        "warrant", "bail", "custody",
        # Legal proceedings
        "judgment", "judgement", "verdict",
        "hearing", "trial", "proceedings",
        "advocate", "lawyer", "attorney",
        "petition", "appeal", "writ"
    ],
    "Semester / Examination": [
        # Academic terms
        "semester", "semesters",
        "exam", "exams", "examination", "examinations",
        "marks", "mark", "grade", "grades",
        "result", "results",
        # Examination-related
        "admit card", "admitcard", "admission card",
        "backlog", "backlogs", "back paper", "backpaper",
        "supplementary", "supplementary exam",
        "re-examination", "reexamination", "re exam",
        # Academic performance
        "cgpa", "sgpa", "gpa",
        "pass", "fail", "failed",
        "promotion", "promoted", "detention",
        # Course-related
        "course", "courses", "subject", "subjects",
        "credit", "credits", "attendance",
        "assignment", "assignments", "project"
    ],
    "Non-Suspension Service": [
        # Service-related
        "transfer", "transferred", "transferring",
        "posting", "posted", "post",
        "promotion", "promoted", "promoting",
        # Service records
        "service book", "servicebook",
        "service record", "service records",
        "pay fixation", "payfixation", "pay-fixation",
        "salary", "increment", "increments",
        # Administrative
        "appointment", "appointed",
        "deputation", "deputed",
        "leave", "leave application", "leave sanction",
        "retirement", "retired", "pension",
        # Other service matters
        "seniority", "seniority list",
        "confirmation", "confirmed",
        "probation", "probationary",
        "regularization", "regularized"
    ]
}

def classify_grievance(text):
    """
    Classify grievance text into categories using improved keyword matching.
    Uses word boundaries and scoring system for better accuracy.
    """
    if not text or not isinstance(text, str):
        return "Miscellaneous"
    
    # Normalize text: lowercase and remove extra whitespace
    text_lower = text.lower()
    text_normalized = re.sub(r'\s+', ' ', text_lower).strip()
    
    # Score each category based on keyword matches
    category_scores = defaultdict(float)
    
    for category, keywords in CATEGORIES.items():
        score = 0
        
        for keyword in keywords:
            # Handle multi-word keywords
            if ' ' in keyword:
                # For multi-word phrases, use word boundaries at start and end
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text_normalized, re.IGNORECASE))
                score += matches * 2  # Multi-word matches are weighted higher
            else:
                # For single words, use word boundaries
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text_normalized, re.IGNORECASE))
                score += matches
        
        category_scores[category] = float(score)
    
    # Find the category with the highest score
    if category_scores:
        max_score = max(category_scores.values())
        
        # Only return a category if it has a meaningful score
        if max_score > 0:
            # If there's a tie, prefer more specific categories
            # Return the first category with max score
            for category in CATEGORIES.keys():
                if category_scores[category] == max_score:
                    return category
    
    return "Miscellaneous"
