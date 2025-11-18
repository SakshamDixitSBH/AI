from pathlib import Path
import extract_msg

from .tfidf_index import index


def _parse_msg(path: Path):
    msg = extract_msg.Message(str(path))

    subject = msg.subject or ""
    sender = msg.sender or ""
    to = msg.to or ""
    cc = msg.cc or ""
    sent_at = msg.date or ""
    body = msg.body or ""

    return {
        "id": str(path),
        "subject": subject,
        "from": sender,
        "to": [a.strip() for a in to.split(";") if a.strip()],
        "cc": [a.strip() for a in cc.split(";") if a.strip()],
        "sent_at": sent_at,
        "body": body,
        "file_path": str(path),
    }


def ingest_msg(path: str) -> int:
    p = Path(path).resolve()

    emails = []
    if p.is_file() and p.suffix.lower() == ".msg":
        emails.append(_parse_msg(p))
    elif p.is_dir():
        for f in sorted(p.glob("*.msg")):
            emails.append(_parse_msg(f))
    else:
        raise ValueError(f"{path} is neither a .msg file nor a folder of .msg files")

    texts = []
    metas = []

    for e in emails:
        header = (
            f"Subject: {e['subject']}\n"
            f"From: {e['from']}\n"
            f"To: {', '.join(e['to'])}\n"
            f"CC: {', '.join(e['cc'])}\n"
            f"Date: {e['sent_at']}\n\n"
        )
        text = header + (e["body"] or "")
        if not text.strip():
            continue

        texts.append(text)
        metas.append(
            {
                "source_type": "email",
                "source": e["file_path"],
                "subject": e["subject"],
                "from": e["from"],
                "sent_at": e["sent_at"],
            }
        )

    if texts:
        index.add_documents(texts, metas)

    return len(texts)
