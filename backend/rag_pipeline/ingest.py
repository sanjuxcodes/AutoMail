import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
RAG_DATA_DIR = BASE_DIR / "rag" / "data"
INDEX_PATH = RAG_DATA_DIR / "faiss.index"
CHUNKS_PATH = RAG_DATA_DIR / "policy_chunks.json"

# Chunking parameters
CHUNK_SIZE_TOKENS = 500
OVERLAP_TOKENS = 50
DEMO_CHUNKS_PATH = RAG_DATA_DIR / "demo_policy_chunks.json"
DEMO_CHUNK_MARKER = "DEMO_POLICY_CHUNK_MARKER"

DEMO_CATEGORY_KEYS: Dict[str, Dict[str, Any]] = {
    "Suspension / Disciplinary": {
        "code": "SUS",
        "keywords": ["suspension", "disciplinary", "inquiry", "charge", "reinstatement", "allowance"],
        "clauses": [
            "Where suspension is ordered, a formal memorandum of charges shall be issued within fifteen working days, failing which the authority shall record written reasons for delay.",
            "No suspension shall ordinarily exceed ninety days without documented review by the competent authority and communication of review outcome to the employee.",
            "Departmental inquiry shall be initiated promptly with appointment of an inquiry officer and presenting officer through a written order.",
            "The suspended employee shall be provided access to relied-upon documents unless disclosure is legally restricted and reasons are recorded in writing.",
            "A clear schedule of inquiry hearings shall be communicated at least seven days in advance to ensure procedural fairness.",
            "Requests for adjournment shall be considered only on justified grounds and shall not be used to prolong proceedings unreasonably.",
            "Subsistence allowance shall be released as per applicable service rules and any delay beyond one salary cycle shall be explained in writing.",
            "If criminal investigation is cited as the basis of suspension, periodic administrative review shall continue independently of trial timelines.",
            "Witness examination and cross-examination records shall be preserved and made available in the case file for appellate review.",
            "Final disciplinary order shall contain findings on each article of charge, evidentiary basis, and reasons for penalty imposed.",
            "Penalty shall be proportionate to proven misconduct and shall consider prior service record and mitigating submissions.",
            "Where charges are not proved, reinstatement shall be processed without avoidable delay and consequential service benefits handled under rules.",
            "Where only part of charges are proved, authority shall issue a reasoned order distinguishing proved and unproved allegations.",
            "Appeal rights, timeline, and appellate authority details shall be explicitly provided in the final order.",
            "Records of suspension, review, inquiry proceedings, and reinstatement decision shall be digitized for institutional accountability.",
            "Institution head shall submit quarterly suspension pendency statement to higher authority for monitoring excessive delays.",
            "In case of prolonged suspension, authority shall consider transfer to non-sensitive duties where permissible under service rules.",
            "Complaints regarding procedural irregularity in disciplinary process shall be examined by designated grievance authority within thirty days.",
        ],
    },
    "FIR / Arrest": {
        "code": "FIR",
        "keywords": ["fir", "arrest", "police", "criminal", "custody", "investigation"],
        "clauses": [
            "Any information regarding FIR or arrest involving institutional personnel shall be reported to the head of institution within twenty-four hours of receipt.",
            "Upon receipt of credible arrest information, the institution shall verify facts through available official channels before issuing administrative orders.",
            "Administrative action shall distinguish between registration of FIR, arrest, remand status, and filing of chargesheet.",
            "Where arrest exceeds forty-eight hours, service consequences shall be processed strictly in accordance with applicable service regulations.",
            "Institution shall nominate a nodal officer for coordination with law enforcement and for maintaining documented communication logs.",
            "No public communication shall disclose sensitive personal data beyond legally permissible limits during ongoing investigation.",
            "Academic and examination continuity measures shall be arranged to avoid disruption where key functionaries are unavailable due to legal proceedings.",
            "If allegations concern campus safety, immediate interim safeguards shall be implemented pending competent authority decision.",
            "Institution shall preserve relevant records and electronic logs that may be sought by lawful investigative agencies.",
            "Legal status updates received from courts or police shall be placed before competent authority for periodic administrative review.",
            "Where acquittal or closure report is received, consequential administrative review shall be initiated within thirty days.",
            "Where chargesheet is filed, disciplinary authority may proceed under service rules independent of criminal trial timetable, subject to legal advice.",
            "Students and staff shall be informed only through authorized notice channels to prevent rumor-driven disruption.",
            "Victim-support and grievance channels shall remain available and confidential during criminal case handling.",
            "Any intimidation, retaliation, or interference with witnesses shall attract immediate disciplinary scrutiny.",
            "Institution shall prepare monthly status report on pending FIR-linked administrative actions for supervisory authority.",
            "Requests for certified copies of institutional records for legal proceedings shall be processed under record-disclosure rules.",
            "All FIR/arrest related files shall include timeline sheet indicating event date, reporting date, action date, and review date.",
        ],
    },
    "Semester / Examination": {
        "code": "EXM",
        "keywords": ["semester", "exam", "result", "recheck", "marks", "correction", "tabulation"],
        "clauses": [
            "Examination schedule, admit-card release, and evaluation timelines shall be published in advance through official institutional channels.",
            "Result publication delay beyond notified timeline shall be accompanied by a formal notice indicating revised date and cause of delay.",
            "Students may apply for rechecking or review within the notified window through prescribed form and fee process.",
            "Rechecking shall verify totaling, unmarked answers, and data transfer integrity without re-evaluation unless specifically permitted by regulations.",
            "Applications for correction of name, roll number, subject code, or paper mismatch shall be processed on priority before final mark-sheet issuance.",
            "Tabulation errors identified post-publication shall be rectified through approval workflow with audit trail.",
            "Provisional certificates may be issued where final documents are delayed, subject to verification of completion status.",
            "Absentee and withheld cases shall be clearly tagged with reasons and expected resolution path in student portal or notice board.",
            "Backlog, supplementary, and special examination notices shall include eligibility, fee, and timeline details.",
            "Internal assessment disputes shall be reviewed by department-level committee with recorded findings.",
            "Examination grievance cells shall acknowledge complaints within three working days and provide disposal timeline.",
            "Digitized scripts or marks data shall be retained for the notified retention period for dispute resolution.",
            "If server or portal outage affects form submission, compensatory submission window shall be provided through official notice.",
            "Late fee waiver in exceptional cases shall be governed by transparent criteria and competent authority approval.",
            "Result correction orders shall be communicated to all affected sections including scholarship and migration desks.",
            "Student requests for transcript and duplicate mark-sheet shall be processed through standardized SLA-based workflow.",
            "Misconduct in examinations shall be processed under unfair-means rules with opportunity of hearing.",
            "Semester progression decisions impacted by pending result disputes shall be handled through interim academic protection measures where permitted.",
        ],
    },
    "Miscellaneous": {
        "code": "MSC",
        "keywords": ["certificate", "transfer", "hostel", "scholarship", "fees", "administrative", "grievance"],
        "clauses": [
            "All miscellaneous administrative grievances shall be acknowledged within three working days with a unique tracking reference.",
            "The concerned section shall provide a reasoned response within fifteen working days unless a longer period is approved with recorded reasons.",
            "Requests for certificates and attested copies shall follow published document checklist and service timelines.",
            "Fee-related corrections, refunds, and ledger mismatches shall be processed through finance verification with auditable entries.",
            "Scholarship verification delays shall be escalated to nodal officer with weekly status update until disposal.",
            "Transfer, migration, and no-objection requests shall be disposed based on eligibility and outstanding dues verification.",
            "Hostel-related grievances on allotment, safety, and facilities shall be reviewed by hostel committee with action minutes.",
            "Accessibility and student-support requests shall be prioritized under inclusive support norms.",
            "Where grievance concerns multiple departments, a lead coordinating officer shall be designated for single-point communication.",
            "All decisions impacting student rights or employee service conditions shall state factual basis and applicable rule position.",
            "If requested records are unavailable, response shall indicate reason, custodian, and next review date.",
            "Digital grievance portal entries shall be mapped to manual register to prevent duplication and closure mismatch.",
            "Escalation matrix and appellate contact details shall be displayed in official notices and portal pages.",
            "Repeated grievances on the same unresolved issue shall be tagged for supervisory review.",
            "Time-bound action taken report shall be generated for grievances marked urgent by competent authority.",
            "Interim relief measures, where permissible, shall be considered to prevent avoidable hardship during final disposal.",
            "Communication to complainant shall remain factual, respectful, and free from speculative statements.",
            "Quarterly analytics of miscellaneous grievances shall be prepared to identify recurring administrative bottlenecks.",
        ],
    },
}


def _map_category(raw_category: str) -> str:
    c = (raw_category or "").strip().lower()
    if "susp" in c or "discip" in c:
        return "Suspension / Disciplinary"
    if "fir" in c or "arrest" in c or "police" in c:
        return "FIR / Arrest"
    if "semester" in c or "exam" in c or "result" in c:
        return "Semester / Examination"
    return "Miscellaneous"


def _extract_common_issue_terms(emails: List[str], limit: int = 5) -> List[str]:
    stop = {
        "the", "and", "for", "that", "with", "from", "this", "have", "has", "are", "was", "were", "you", "your",
        "regarding", "please", "kindly", "subject", "email", "institution", "college", "department", "request",
    }
    freq: Dict[str, int] = {}
    for mail in emails:
        for tok in re.findall(r"[a-zA-Z]{4,}", str(mail).lower()):
            if tok in stop:
                continue
            freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:limit]]


def generate_demo_policy_dataset(
    grievances_excel_path: str,
    output_path: str = str(DEMO_CHUNKS_PATH),
    min_emails_per_category: int = 20,
    max_emails_per_category: int = 50,
) -> Dict[str, Any]:
    """Generate high-quality category-aligned demo policy corpus from grievances.xlsx."""
    excel_path = Path(grievances_excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Missing grievances file: {excel_path}")

    import pandas as pd

    frame = pd.read_excel(excel_path)
    content_col = "content" if "content" in frame.columns else "email_content"
    if content_col not in frame.columns or "category" not in frame.columns:
        raise ValueError("grievances.xlsx must contain category and content/email_content columns")

    records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"categories": {}, "total_clauses": 0}

    canonical_categories = [
        "Suspension / Disciplinary",
        "FIR / Arrest",
        "Semester / Examination",
        "Miscellaneous",
    ]

    frame = frame.copy()
    frame["_mapped_category"] = frame["category"].fillna("").astype(str).apply(_map_category)

    for mapped_cat in canonical_categories:
        cat_rows = frame[frame["_mapped_category"] == mapped_cat]
        sample_rows = cat_rows.head(max_emails_per_category)
        emails = [str(v) for v in sample_rows[content_col].fillna("").tolist() if str(v).strip()]

        # If category has too few rows, still generate the baseline policy bank for demo reliability.
        common_terms = _extract_common_issue_terms(emails, limit=5)
        cfg = DEMO_CATEGORY_KEYS[mapped_cat]
        code = cfg["code"]
        clauses = list(cfg["clauses"])

        if common_terms:
            terms_txt = ", ".join(common_terms)
            clauses.insert(
                0,
                f"For recurring issues identified in grievance communications ({terms_txt}), the institution shall maintain a time-bound action tracker and provide category-specific disposal updates.",
            )

        # Keep strict 15-25 clause range.
        clauses = clauses[:25]
        if len(clauses) < 15:
            raise ValueError(f"Insufficient demo clauses generated for category: {mapped_cat}")

        for i, text in enumerate(clauses, start=1):
            records.append(
                {
                    "policy_id": "DEMO_POLICY",
                    "section_title": mapped_cat,
                    "clause_id": f"{code}-{i:02d}",
                    "text": text,
                }
            )

        summary["categories"][mapped_cat] = {
            "source_emails_analyzed": int(min(len(emails), max_emails_per_category)),
            "requested_min": int(min_emails_per_category),
            "clauses_generated": int(len(clauses)),
            "top_issue_terms": common_terms,
            "coverage_status": "ok" if len(emails) >= min_emails_per_category else "low_source_volume",
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    summary["total_clauses"] = len(records)
    summary["output_path"] = str(out)
    return summary


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens with overlap.
    
    Args:
        text: Input text to split
        chunk_size: Target chunk size in tokens
        overlap: Number of tokens to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split by sentences first
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        # Rough token count (approximation)
        sentence_tokens = len(sentence.split())
        
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Save current chunk and start new one
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Keep last few sentences for overlap
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_chunk):
                s_tokens = len(s.split())
                if overlap_tokens + s_tokens <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_tokens = overlap_tokens + sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
    
    return chunks


def split_marker_chunks(text: str, marker: str = DEMO_CHUNK_MARKER) -> List[str]:
    """Split text into exact chunks when marker tags are present in source documents."""
    if marker not in text:
        return []

    raw_parts = text.split(marker)[1:]
    chunks: List[str] = []
    for part in raw_parts:
        normalized = " ".join(part.split()).strip()
        if normalized:
            chunks.append(normalized)
    return chunks


def split_numbered_clause_chunks(text: str) -> List[str]:
    """Split text into chunks when clauses are prefixed as `Clause N:`."""
    pattern = re.compile(r"(?:^|\n)\s*Clause\s+[A-Za-z0-9_-]+\s*:\s*", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    chunks: List[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        normalized = " ".join(text[start:end].split()).strip()
        if normalized:
            chunks.append(normalized)

    return chunks


def update_faiss_index(new_embeddings: np.ndarray, existing_index_path: Path = INDEX_PATH) -> None:
    """
    Update FAISS index by appending new embeddings to existing index.
    
    Args:
        new_embeddings: New embeddings to add (shape: [N, 384] for MiniLM)
        existing_index_path: Path to existing FAISS index
    """
    if new_embeddings.size == 0:
        return
    
    new_embeddings = np.asarray(new_embeddings).astype("float32")
    
    if existing_index_path.exists():
        # Load existing index and append new embeddings
        index = faiss.read_index(str(existing_index_path))
        index.add(new_embeddings)
    else:
        # Create new index
        dim = new_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(new_embeddings)
    
    # Save updated index
    faiss.write_index(index, str(existing_index_path))


def ingest_policy_document(file_path: str, policy_id: Optional[str] = None, section_title: str = "General") -> Dict[str, Any]:
    """
    Ingest a policy document (PDF or TXT), chunk it, generate embeddings, and update index.
    
    Args:
        file_path: Path to the policy document
        policy_id: Optional policy identifier; defaults to filename
        section_title: Section title for metadata
    
    Returns:
        Dictionary with ingestion statistics
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Policy document not found: {file_path}")
    
    # Extract text from document
    if file_path.suffix.lower() == ".pdf":
        text = ""
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception:
            # Fallback for PDFs that parse better with PyMuPDF.
            try:
                import fitz  # type: ignore

                doc = fitz.open(file_path)
                text = "\n".join(page.get_text() for page in doc)
            except ImportError:
                raise ImportError(
                    "PDF extraction failed with pdfplumber and PyMuPDF is unavailable. Install: pip install pymupdf"
                )
    elif file_path.suffix.lower() in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    if not text or not text.strip():
        raise ValueError(f"No text extracted from: {file_path}")
    
    # Determine policy_id
    if policy_id is None:
        policy_id = file_path.stem
    
    # Split into chunks. If marker-delimited chunks exist, preserve them exactly.
    chunks = split_marker_chunks(text)
    if not chunks and file_path.parent.name.lower() == "demo_policies":
        chunks = split_numbered_clause_chunks(text)
    if not chunks:
        chunks = split_text_into_chunks(text, chunk_size=CHUNK_SIZE_TOKENS, overlap=OVERLAP_TOKENS)
    if not chunks:
        raise ValueError(f"No chunks generated from: {file_path}")
    
    # Load or initialize policy_chunks.json
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []
    
    # Generate metadata and create chunk objects
    max_clause_id = 0
    if existing_chunks:
        max_clause_id = max(int(c.get("clause_id", 0)) for c in existing_chunks if c.get("clause_id"))
    
    new_chunks = []
    for idx, chunk_text in enumerate(chunks, start=1):
        clause_id = max_clause_id + idx
        new_chunks.append(
            {
                "chunk_id": clause_id,
                "text": chunk_text,
                "policy_id": policy_id,
                "section_title": section_title,
                "clause_id": clause_id,
                "source_file": str(file_path),
            }
        )
    
    # Generate embeddings for new chunks
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    texts_to_embed = [chunk["text"] for chunk in new_chunks]
    embeddings = embedder.encode(
        texts_to_embed,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings).astype("float32")
    
    # Update FAISS index
    update_faiss_index(embeddings)
    
    # Append new chunks to policy_chunks.json
    all_chunks = existing_chunks + new_chunks
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    return {
        "success": True,
        "policy_id": policy_id,
        "chunks_added": len(new_chunks),
        "total_chunks": len(all_chunks),
        "embeddings_generated": embeddings.shape[0],
        "index_path": str(INDEX_PATH),
        "chunks_path": str(CHUNKS_PATH),
    }


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        default_excel = BASE_DIR / "data" / "grievances.xlsx"
        excel_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else default_excel
        summary = generate_demo_policy_dataset(str(excel_arg), str(DEMO_CHUNKS_PATH))
        print("Demo policy dataset generated")
        print(json.dumps(summary, indent=2))
        return

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        policy_chunks = json.load(f)

    if not policy_chunks:
        raise ValueError("policy_chunks.json is empty")

    texts = [chunk.get("text", "") for chunk in policy_chunks if chunk.get("text")]
    if not texts:
        raise ValueError("No valid text fields found in policy chunks")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    print("Ingestion complete")
    print(f"Chunks indexed: {len(texts)}")
    print(f"Embedding dim: {dim}")
    print(f"Index saved at: {INDEX_PATH}")


if __name__ == "__main__":
    main()
