"""Minimal PST ingest using pypff.

Note:
  - Requires the `pypff` package and native libpff installed on your system.
  - If that's painful, consider exporting PST to .eml/.mbox and parsing with stdlib instead.
"""
import uuid
from pathlib import Path
from typing import List, Dict, Any

try:
    import pypff  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pypff = None

from .vector_store import get_collection

def _message_to_email_dict(msg) -> Dict[str, Any]:
    """Extract minimal fields from a pypff message object."""
    subject = msg.subject or ""
    sender = msg.sender_name or msg.sender_email_address or ""
    to = msg.display_to or ""
    cc = msg.display_cc or ""
    body = msg.plain_text_body or msg.html_body or ""
    sent = msg.client_submit_time or ""
    conversation_id = msg.conversation_topic or subject

    return {
        "id": msg.identifier,
        "subject": subject,
        "from": sender,
        "to": [a.strip() for a in to.split(";") if a.strip()],
        "cc": [a.strip() for a in cc.split(";") if a.strip()],
        "sent_at": str(sent),
        "body": body,
        "thread_id": conversation_id,
    }

def _walk_folder(folder, collected: List[Dict[str, Any]]):
    # messages
    for i in range(folder.number_of_messages):
        m = folder.get_message(i)
        collected.append(_message_to_email_dict(m))
    # subfolders
    for j in range(folder.number_of_sub_folders):
        sub = folder.get_sub_folder(j)
        _walk_folder(sub, collected)

def load_pst_emails(pst_path: str) -> List[Dict[str, Any]]:
    """Return a list of normalized email dicts from a PST file."""
    if pypff is None:
        raise RuntimeError("pypff is not installed. Install it to use PST ingestion.")

    pst_path = str(Path(pst_path).resolve())
    file = pypff.file()
    file.open(pst_path)
    root = file.get_root_folder()
    emails: List[Dict[str, Any]] = []
    _walk_folder(root, emails)
    file.close()
    return emails

def ingest_pst(pst_path: str) -> int:
    """Ingest PST messages into Chroma as source_type='email'. Returns count."""
    col = get_collection()
    emails = load_pst_emails(pst_path)
    pst_path = str(Path(pst_path).resolve())

    ids, docs, metas = [], [], []
    for e in emails:
        text_lines = [
            f"Subject: {e.get('subject','')}",
            f"From: {e.get('from','')}",
            f"To: {', '.join(e.get('to', []))}",
            f"CC: {', '.join(e.get('cc', []))}",
            f"Date: {e.get('sent_at','')}",
            "",
            e.get("body", ""),
        ]
        text = "\n".join(text_lines).strip()
        if not text:
            continue

        ids.append(str(uuid.uuid4()))
        docs.append(text)
        metas.append({
            "source_type": "email",
            "source": pst_path,
            "message_id": e.get("id"),
            "thread_id": e.get("thread_id"),
            "subject": e.get("subject"),
            "from": e.get("from"),
            "sent_at": e.get("sent_at"),
        })

    if docs:
        col.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)
