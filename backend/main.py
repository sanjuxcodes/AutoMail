import email
from itertools import count
import os
import pandas as pd
import re
import json
import ast
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict
from email_parser import parse_eml
from classifier import classify_grievance
from followup import detect_followup
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from chatbot import ask
from priority_llm import get_llm_priority, apply_thread_escalation
from pydantic import BaseModel
from rag.reply_generator import generate_suggested_reply_with_evidence
from rag.retriever import reload_retriever_data
from rag_pipeline.ingest import ingest_policy_document, CHUNKS_PATH, INDEX_PATH
import logging
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILES_DIR = "./files"
EXCEL_FILE = "./data/grievances.xlsx"
POLICY_DIR = Path(__file__).parent / "policies"
POLICY_DIR.mkdir(parents=True, exist_ok=True)
DEMO_EMAILS_PATH = Path(__file__).parent / "demo_emails_from_chunks.json"
DEMO_POLICY_DIR = Path(__file__).parent / "demo_policies"
DEMO_POLICY_DIR.mkdir(parents=True, exist_ok=True)

# email source mode: demo | files | dynamic
EMAIL_SOURCE_MODE = os.getenv("EMAIL_SOURCE_MODE", "demo").strip().lower()
DEMO_USE_LLM_PRIORITY = os.getenv("DEMO_USE_LLM_PRIORITY", "false").strip().lower() == "true"

# Load college database
COLLEGES_DB = []
try:
    colleges_path = Path(__file__).parent / "data" / "colleges.json"
    with open(colleges_path, 'r', encoding='utf-8') as f:
        COLLEGES_DB = json.load(f)
    print(f"Loaded {len(COLLEGES_DB)} colleges from database")
except Exception as e:
    print(f"Warning: Could not load colleges database: {e}")

COLUMNS = [
    "email_id", "parent_email_id", "sender", "subject",
    "category", "mail_type", "followup_count","priority",
    "date", "content", "attachments", "eml_file"
]

def clean_for_excel(text):
    """Remove illegal characters that Excel cannot handle"""
    if not isinstance(text, str):
        return text
    # Remove control characters (0x00-0x1F) except tab, newline, carriage return
    return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)


def _generate_email_id(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def _heuristic_priority(text: str, subject: str) -> int:
    text_lower = f"{subject} {text}".lower()
    if any(k in text_lower for k in ["chief secretary", "secretary higher education", "court order", "legal notice"]):
        return 1
    if any(k in text_lower for k in ["fir", "arrest", "police", "misconduct", "suspension", "disciplinary"]):
        return 2
    if any(k in text_lower for k in ["salary", "fund release", "increment", "arrear"]):
        return 4
    if any(k in text_lower for k in ["exam", "result", "admission", "deadline", "correction", "scholarship"]):
        return 5
    return 6


def _load_demo_emails() -> list[Dict[str, Any]]:
    if not DEMO_EMAILS_PATH.exists():
        raise FileNotFoundError(f"Missing demo email source: {DEMO_EMAILS_PATH}")

    with open(DEMO_EMAILS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("demo_emails_from_chunks.json must contain a list of email objects")

    return payload


def _build_demo_dataframe() -> pd.DataFrame:
    demo_emails = _load_demo_emails()
    df = pd.DataFrame()
    base_date = datetime.now() - timedelta(days=max(len(demo_emails), 1))

    for i, item in enumerate(demo_emails):
        subject = str(item.get("subject", "")).strip()
        body = str(item.get("body", "")).strip()
        if not subject and not body:
            continue

        content = f"{subject}\n{body}".strip()
        email_id = _generate_email_id(content)
        email_date = (base_date + timedelta(days=i)).date().isoformat()
        institute_name = COLLEGES_DB[i % len(COLLEGES_DB)] if COLLEGES_DB else "Unknown Institute"

        sender_name = "Grievance Applicant"
        sender_email = f"demo_applicant_{i + 1}@example.edu"

        category = classify_grievance(content)

        base_priority = None
        if DEMO_USE_LLM_PRIORITY:
            try:
                base_priority = get_llm_priority(content, subject, sender_name)
            except Exception:
                base_priority = None
        if base_priority is None:
            base_priority = _heuristic_priority(content, subject)

        mock_email_obj = {
            "email_id": email_id,
            "parent_email_id": "",
            "sender": sender_name,
            "subject": subject,
            "content": content,
            "date": email_date,
        }

        mail_type, parent_id, followup_count = detect_followup(mock_email_obj, df)
        final_priority = apply_thread_escalation(base_priority, followup_count)
        if final_priority is None:
            final_priority = base_priority

        row = {
            "email_id": email_id,
            "parent_email_id": parent_id or "",
            "sender": sender_name,
            "sender_email": sender_email,
            "institute_name": institute_name,
            "subject": subject,
            "category": category,
            "mail_type": mail_type,
            "followup_count": followup_count,
            "priority": int(final_priority),
            "date": email_date,
            "content": content,
            "attachments": "[]",
            "eml_file": f"demo_{i + 1}.eml",
            "is_demo": True,
        }

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    return df


def _build_files_dataframe() -> pd.DataFrame:
    if not os.path.isdir(FILES_DIR):
        return pd.DataFrame(columns=COLUMNS)

    eml_files = [f for f in os.listdir(FILES_DIR) if f.endswith(".eml")]
    total_files = len(eml_files)
    processed = 0
    df = pd.DataFrame(columns=COLUMNS)

    print(f"📧 Found {total_files} email files to process...")

    for file in eml_files:
        processed += 1
        print(f"[{processed}/{total_files}] Processing: {file}")

        parsed_email = parse_eml(os.path.join(FILES_DIR, file))
        if parsed_email is None:
            print(f"⚠️ Skipping {file} (parse error)")
            continue

        if not df.empty and parsed_email["email_id"] in df["email_id"].values:
            print("✓ Already processed")
            continue

        category = classify_grievance(parsed_email["content"])
        mail_type, _, followup_count = detect_followup(parsed_email, df)

        try:
            base_priority = get_llm_priority(
                parsed_email["content"],
                parsed_email.get("subject", ""),
                parsed_email.get("sender", ""),
            )
        except Exception:
            base_priority = None
        if base_priority is None:
            base_priority = _heuristic_priority(parsed_email.get("content", ""), parsed_email.get("subject", ""))

        final_priority = apply_thread_escalation(base_priority, followup_count)
        if final_priority is None:
            final_priority = base_priority

        row = {
            **parsed_email,
            "category": category,
            "mail_type": mail_type,
            "followup_count": followup_count,
            "priority": final_priority,
            "date": parsed_email.get("date") or datetime.now().strftime("%Y-%m-%d"),
            "is_demo": False,
        }

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        print("✓ Added to database")

    return df


def _collect_policy_pdf_files(*directories: Path) -> list[Path]:
    pdf_candidates: list[Path] = []
    for directory in directories:
        pdf_candidates.extend(list(directory.glob("*.pdf")) + list(directory.glob("*.PDF")))
    return sorted({p.resolve() for p in pdf_candidates})


def _sync_policy_pdfs_from_disk(*, include_demo: bool = True, include_uploaded: bool = True) -> Dict[str, Any]:
    """Rebuild the policy corpus from PDFs stored on disk."""
    directories: list[Path] = []
    if include_uploaded:
        directories.append(POLICY_DIR)
    if include_demo:
        directories.append(DEMO_POLICY_DIR)

    pdf_files = _collect_policy_pdf_files(*directories)

    if not pdf_files:
        return {
            "pdf_count": 0,
            "indexed_count": 0,
            "chunks_path": str(CHUNKS_PATH),
            "index_path": str(INDEX_PATH),
        }

    if CHUNKS_PATH.exists():
        CHUNKS_PATH.unlink()
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    generated_any = False
    for pdf_path in pdf_files:
        try:
            section_title = "Demo Policy" if pdf_path.parent == DEMO_POLICY_DIR else "General"
            ingest_policy_document(
                file_path=str(pdf_path),
                policy_id=pdf_path.stem,
                section_title=section_title,
            )
            generated_any = True
        except Exception as exc:
            print(f"[WARN] Failed to ingest policy PDF {pdf_path.name}: {exc}")

    if not generated_any:
        return {
            "pdf_count": len(pdf_files),
            "indexed_count": 0,
            "chunks_path": str(CHUNKS_PATH),
            "index_path": str(INDEX_PATH),
        }

    reload_retriever_data(force=True)
    return {
        "pdf_count": len(pdf_files),
        "indexed_count": len(pdf_files),
        "chunks_path": str(CHUNKS_PATH),
        "index_path": str(INDEX_PATH),
    }


def _sync_demo_policy_pdfs(force_index_rebuild: bool = False) -> Dict[str, Any]:
    """Build the demo policy corpus from PDFs placed in backend/demo_policies."""
    return _sync_policy_pdfs_from_disk(include_demo=True, include_uploaded=False)

def run(source_mode_override: str | None = None):
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(EXCEL_FILE), exist_ok=True)

    source_mode = (source_mode_override or EMAIL_SOURCE_MODE).strip().lower()

    # Ensure the active corpus reflects PDFs on disk before processing emails.
    if source_mode == "demo":
        _sync_policy_pdfs_from_disk(include_demo=True, include_uploaded=True)

    if source_mode == "demo":
        print("📦 Email source mode: demo (from demo_emails_from_chunks.json)")
        df = _build_demo_dataframe()
    elif source_mode == "files":
        print("📂 Email source mode: files (from backend/files/*.eml)")
        df = _build_files_dataframe()
    elif source_mode == "dynamic":
        print("🔌 Email source mode: dynamic (placeholder, falling back to demo source)")
        df = _build_demo_dataframe()
    else:
        raise ValueError(f"Unsupported EMAIL_SOURCE_MODE: {source_mode}")

    # Clean all text fields for Excel compatibility
    print("\n🧹 Cleaning data for Excel compatibility...")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(clean_for_excel)
    
    df.to_excel(EXCEL_FILE, index=False)
    print(f"\n✅ Processing complete! Total records: {len(df)}")
    print(f"📊 Data saved to: {EXCEL_FILE}")

# ---------------- FASTAPI APP ----------------
app = FastAPI()

# Enable CORS for frontend
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- FASTAPI APP ----------------


@app.on_event("startup")
def _bootstrap_demo_mode() -> None:
    synced = _sync_policy_pdfs_from_disk(include_demo=True, include_uploaded=True)
    if synced["pdf_count"]:
        print("[INFO] Policy corpus synced from PDFs into policy_chunks.json for RAG retrieval.")

@app.get("/")
def root():
    return {"status": "Backend running on Hugging Face"}

@app.get("/health")
def health_check():
    """Health check endpoint for Docker/load balancer"""
    return {"status": "healthy", "service": "AI Grievance Management Backend"}
    
@app.get("/categories")
def get_categories():
    """Get all categories with count"""
    if not os.path.exists(EXCEL_FILE):
        return []
    
    df = pd.read_excel(EXCEL_FILE)
    categories = df.groupby("category").size().reset_index(name="count")
    
    return [
        {"name": row["category"], "count": int(row["count"])}
        for _, row in categories.iterrows()
    ]

@app.get("/colleges")
def get_colleges():
    """Get list of all colleges in database"""
    return {"colleges": COLLEGES_DB, "total": len(COLLEGES_DB)}


@app.get("/emails")
def get_emails(
    category: str = Query(..., description="Email category or 'all' for all categories"),
    is_demo: str = Query(None, description="Filter by is_demo: 'true', 'false', or None for all")
):
    """Get all emails for a specific category, with optional is_demo filtering"""
    try:
        print(f"Received request for category: '{category}', is_demo: {is_demo}")

        if not os.path.exists(EXCEL_FILE):
            print(f"Excel file not found: {EXCEL_FILE}")
            return []

        df = pd.read_excel(EXCEL_FILE)
        print(f"Total records in Excel: {len(df)}")
        print(f"Unique categories: {df['category'].unique().tolist()}")

        # Backward compatibility for previously generated datasets.
        if "priority" not in df.columns:
            df["priority"] = 99
        if "followup_count" not in df.columns:
            df["followup_count"] = 0
        if "parent_email_id" not in df.columns:
            df["parent_email_id"] = ""
        if "attachments" not in df.columns:
            df["attachments"] = "[]"
        if "content" not in df.columns:
            df["content"] = ""
        if "eml_file" not in df.columns:
            df["eml_file"] = ""
        if "is_demo" not in df.columns:
            df["is_demo"] = False
        df["priority"] = df["priority"].fillna(99)  # ← add this
        # Filtering
        filtered = df.copy()
        if category != "all":
            filtered = filtered[filtered["category"] == category]
        if is_demo is not None:
            if is_demo.lower() == "true":
                filtered = filtered[filtered["is_demo"] == True]
            elif is_demo.lower() == "false":
                filtered = filtered[(filtered["is_demo"] == False) | (filtered["is_demo"].isna())]

        filtered = filtered.sort_values(by="priority", ascending=True)
        print(f"Found {len(filtered)} emails for category '{category}' and is_demo={is_demo}")

        result = [
            {
                "id": str(row["email_id"]) if pd.notna(row["email_id"]) else "",
                "sender": str(row["sender"]),
                "senderEmail": str(row.get("sender_email", "student@example.edu")),
                "instituteName": str(row.get("institute_name", "Unknown Institute")),
                "subject": str(row["subject"]),
                "content": str(row["content"]) if pd.notna(row["content"]) and str(row["content"]) != "nan" else "",
                "hasAttachment": len(ast.literal_eval(str(row["attachments"]))) > 0 if pd.notna(row["attachments"]) and str(row["attachments"]) not in ["nan", "", "[]"] else False,
                "attachments": ast.literal_eval(str(row["attachments"])) if pd.notna(row["attachments"]) and str(row["attachments"]) not in ["nan", "", "[]"] else [],
                "emlFile": str(row["eml_file"]) if pd.notna(row["eml_file"]) and str(row["eml_file"]) != "nan" else "",
                "category": str(row["category"]),
                "mailType": str(row["mail_type"]).lower() if pd.notna(row["mail_type"]) else "fresh",
                "followUpCount": int(row["followup_count"]) if pd.notna(row["followup_count"]) else 0,
                "priority": int(row["priority"]) if pd.notna(row["priority"]) else 99,
                "emailDate": str(row["date"]),
                "parentEmailId": str(row["parent_email_id"]) if pd.notna(row["parent_email_id"]) else "",
                "is_demo": bool(row["is_demo"]) if "is_demo" in row else False
            }
            for _, row in filtered.iterrows()
        ]

        return result
    except Exception as e:
        print(f"Error in get_emails: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/process")
def process_emails():
    run()
    return {"status": "emails processed", "source_mode": EMAIL_SOURCE_MODE}


@app.post("/process-demo")
def process_demo_emails():
    """Force demo-json population regardless of configured default source mode."""
    run(source_mode_override="demo")
    return {"status": "demo emails processed", "source_mode": "demo"}


class DemoPolicySyncResponse(BaseModel):
    success: bool
    message: str
    stats: Dict[str, Any]


@app.post("/policies/load-demo", response_model=DemoPolicySyncResponse)
def load_demo_policies():
    """Rebuild policy_chunks.json and FAISS from the demo policy PDFs."""
    try:
        stats = _sync_demo_policy_pdfs(force_index_rebuild=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load demo policies: {exc}")

    return DemoPolicySyncResponse(
        success=True,
        message="Demo policy PDFs loaded and indexed successfully",
        stats=stats,
    )

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    print("📥 CHAT MESSAGE:", repr(request.message))

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    answer = ask(request.message)

    print("🤖 CHAT ANSWER:", repr(answer))
    return {"answer": answer}

@app.get("/dashboard")
def dashboard():
    if not os.path.exists(EXCEL_FILE):
        return {"total": 0, "fresh": 0, "followup": 0}
    
    df = pd.read_excel(EXCEL_FILE)
    
    return {
        "total": len(df),
        "fresh": int((df["mail_type"] == "Fresh").sum()),
        "followup": int((df["mail_type"] == "Follow-up").sum()),
    }

@app.get("/download/attachment/{filename}")
def download_attachment(filename: str):
    """Download an actual attachment file (PDF, DOCX, Excel, etc.)"""
    file_path = os.path.join("./attachments", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Attachment not found")
    
    # Determine media type based on extension
    ext = filename.lower().split('.')[-1]
    media_types = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'xls': 'application/vnd.ms-excel',
        'txt': 'text/plain',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
    }
    media_type = media_types.get(ext, 'application/octet-stream')
    
    # Extract original filename (remove hash prefix)
    original_name = '_'.join(filename.split('_')[1:]) if '_' in filename else filename
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=original_name
    )

@app.get("/download/email/{eml_filename}")
def download_eml(eml_filename: str):
    """Download the original .eml file if needed"""
    file_path = os.path.join(FILES_DIR, eml_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Email file not found")
    return FileResponse(
        file_path,
        media_type="message/rfc822",
        filename=eml_filename
    )


class PolicyUploadResponse(BaseModel):
    success: bool
    message: str
    stats: Dict[str, Any]


def _slugify_policy_id(raw_name: str) -> str:
    base = Path(raw_name).stem if raw_name else "policy"
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_")
    return base or "policy"


def _load_policy_chunks() -> list[Dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        return []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _save_policy_chunks(chunks: list[Dict[str, Any]]) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def _rebuild_index_from_chunks(chunks: list[Dict[str, Any]]) -> None:
    texts = [chunk.get("text", "") for chunk in chunks if chunk.get("text")]

    if not texts:
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))


def _remove_policy_chunks(policy_id: str) -> tuple[list[Dict[str, Any]], int]:
    chunks = _load_policy_chunks()
    retained = [c for c in chunks if str(c.get("policy_id", "")) != policy_id]
    removed = len(chunks) - len(retained)

    if removed:
        _save_policy_chunks(retained)
        _rebuild_index_from_chunks(retained)

    return retained, removed


def _policy_file_exists(policy_id: str) -> bool:
    return _resolve_policy_file_path(policy_id) is not None


def _resolve_policy_file_path(policy_id: str) -> Path | None:
    uploaded_path = POLICY_DIR / f"{policy_id}.pdf"
    if uploaded_path.exists():
        return uploaded_path

    demo_path = DEMO_POLICY_DIR / f"{policy_id}.pdf"
    if demo_path.exists():
        return demo_path

    return None


@app.post("/policies/upload", response_model=PolicyUploadResponse)
async def upload_policy(file: UploadFile = File(...)):
    """Upload a PDF policy and ingest it into policy_chunks.json + FAISS index."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    policy_id = _slugify_policy_id(file.filename)
    policy_path = POLICY_DIR / f"{policy_id}.pdf"
    was_existing = policy_path.exists()

    try:
        with open(policy_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    finally:
        await file.close()

    try:
        # Keep one canonical entry per policy_id by replacing previous chunks on update.
        _remove_policy_chunks(policy_id)

        stats = ingest_policy_document(
            file_path=str(policy_path),
            policy_id=policy_id,
            section_title="General",
        )
        stats["is_update"] = was_existing
        stats["policy_id"] = policy_id

        reload_retriever_data(force=True)

        return PolicyUploadResponse(
            success=True,
            message="Policy uploaded and indexed successfully",
            stats=stats,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest policy: {str(e)}")


@app.get("/policies")
def list_policies():
    """List policies currently indexed in RAG with per-policy chunk counts."""
    chunks = _load_policy_chunks()
    by_policy: Dict[str, int] = {}

    for chunk in chunks:
        pid = str(chunk.get("policy_id", "")).strip()
        if pid:
            by_policy[pid] = by_policy.get(pid, 0) + 1

    policies = []
    for policy_id, chunk_count in sorted(by_policy.items(), key=lambda x: x[0].lower()):
        file_exists = _policy_file_exists(policy_id)
        policies.append(
            {
                "id": policy_id,
                "chunk_count": chunk_count,
                "status": "indexed" if file_exists else "indexed_no_file",
            }
        )

    return {"policies": policies}


@app.get("/policies/{policy_id}")
def get_policy_pdf(policy_id: str):
    """Download/view original policy PDF by policy_id."""
    policy_file = _resolve_policy_file_path(policy_id)
    if policy_file is None:
        raise HTTPException(status_code=404, detail="Policy file not found")

    return FileResponse(
        str(policy_file),
        media_type="application/pdf",
        filename=policy_file.name,
    )


@app.delete("/policies/{policy_id}")
def delete_policy(policy_id: str):
    """Delete a policy PDF and all associated chunks, then rebuild FAISS."""
    policy_file = POLICY_DIR / f"{policy_id}.pdf"
    had_file = policy_file.exists()
    if policy_file.exists():
        policy_file.unlink()

    _, removed_chunks = _remove_policy_chunks(policy_id)
    reload_retriever_data(force=True)

    if removed_chunks == 0 and not had_file:
        raise HTTPException(status_code=404, detail="Policy not found")

    return {
        "success": True,
        "message": f"Policy '{policy_id}' removed",
        "removed_chunks": removed_chunks,
    }

# Request model for reply generation
class GenerateReplyRequest(BaseModel):
    email_content: str
    email_subject: str = ""
    sender: str = ""
    category: str = ""


class GenerateReplyResponse(BaseModel):
    suggested_reply: str
    generation_mode: str = "simple_stabilized"
    status: str = "ok"
    evidence: list[Dict[str, Any]] = []
    trust_score: float | None = None
    trust_label: str | None = None
    trust_visibility: Dict[str, Any] = {}
    retrieval: Dict[str, Any] = {}
    grounding_report: Dict[str, Any] = {}
    sentence_evidence: list[Dict[str, Any]] = []
    explainability: Dict[str, Any] = {}
    internal_view: Dict[str, Any] = {}
    external_view: Dict[str, Any] = {}

@app.post("/generate-reply")
def generate_reply(request: GenerateReplyRequest) -> GenerateReplyResponse:
    """Generate AI-powered suggested reply using RAG system"""
    try:
        logger.info(
            f"Generating reply | sender={request.sender} | subject={request.email_subject}"
        )

        reply_bundle = generate_suggested_reply_with_evidence(
            email_content=request.email_content,
            email_subject=request.email_subject,
            sender=request.sender,
            category=request.category,
        )
        return GenerateReplyResponse(**reply_bundle)
    except Exception as e:
        print(f"Error generating reply: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate reply: {str(e)}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import sys
    import uvicorn

    # If script is run directly, process emails first
    if len(sys.argv) > 1 and sys.argv[1] == "process":
        run()
    else:
        port = int(os.environ.get("PORT", 8000))
        print("\n🚀 Starting FastAPI server...")
        print(f"📍 Backend running at: http://localhost:{port}")
        print(f"📊 API Documentation at: http://localhost:{port}/docs")
        print("\nPress Ctrl+C to stop the server\n")
        uvicorn.run(app, host="0.0.0.0", port=port)
