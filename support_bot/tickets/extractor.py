"""Ticket extraction engine from conversation history and user messages."""

import re
from typing import Any, Dict, List, Optional, Tuple
from support_bot.tickets.models import ExtractionResult


# Known product catalog keywords
COURSE_CATALOG = [
    "python", "fullstack", "data science", "machine learning", "ai bootcamp",
    "web development", "react", "javascript", "devops", "cloud computing"
]

# Order / Enrollment ID regex patterns (e.g. ORD-12345, ENR-98765, PAY-4412, or 5-9 digit numbers)
ID_PATTERNS = [
    re.compile(r"\b(ORD-[A-Z0-9]{4,12})\b", re.IGNORECASE),
    re.compile(r"\b(ENR-[A-Z0-9]{4,12})\b", re.IGNORECASE),
    re.compile(r"\b(PAY-[A-Z0-9]{4,12})\b", re.IGNORECASE),
    re.compile(r"\b(INV-[A-Z0-9]{4,12})\b", re.IGNORECASE),
    re.compile(r"\b(TXN-[A-Z0-9]{4,12})\b", re.IGNORECASE),
    re.compile(r"\b(order\s*(?:id|#|no\.?)?\s*[:=]?\s*)([A-Z0-9-]{4,15})\b", re.IGNORECASE),
    re.compile(r"\b(enrollment\s*(?:id|#|no\.?)?\s*[:=]?\s*)([A-Z0-9-]{4,15})\b", re.IGNORECASE),
    re.compile(r"\b([0-9]{5,10})\b"),
]

# Contact info patterns
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b")


def extract_ticket_data(
    messages: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ExtractionResult:
    """Extract ticket fields from conversation messages.
    
    Checks config['tickets']['mandatory_fields']. If any mandatory field is missing,
    returns missing_fields and a clarification prompt WITHOUT guessing or inferring.
    """
    cfg = config or {}
    ticket_cfg = cfg.get("tickets", {})
    mandatory = ticket_cfg.get(
        "mandatory_fields",
        ["customer_name", "order_or_enrollment_id", "product_or_course", "issue_type", "description"]
    )

    combined_user_text = []
    evidence_refs = []
    customer_name = None
    contact_info = None
    order_id = None
    product_course = None
    issue_type = None

    for m in messages:
        text = m.get("text", "")
        sender = m.get("sender", "user")
        msg_id = m.get("id") or m.get("message_id")
        if msg_id:
            evidence_refs.append(str(msg_id))

        if sender in ("user", "customer"):
            combined_user_text.append(text)

            # Check for customer name self-introduction: "My name is John", "I am Alice"
            if not customer_name:
                name_match = re.search(r"\b(?:my name is|i am|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text, re.IGNORECASE)
                if name_match:
                    customer_name = name_match.group(1).strip()

            # Check email
            if not contact_info:
                email_match = EMAIL_PATTERN.search(text)
                if email_match:
                    contact_info = email_match.group(0).strip()
                else:
                    phone_match = PHONE_PATTERN.search(text)
                    if phone_match:
                        contact_info = phone_match.group(0).strip()

            # Check order / enrollment ID
            if not order_id:
                for pat in ID_PATTERNS:
                    match = pat.search(text)
                    if match:
                        order_id = match.group(match.lastindex or 0).strip()
                        break

            # Check product / course
            if not product_course:
                for c in COURSE_CATALOG:
                    if re.search(r"\b" + re.escape(c) + r"\b", text, re.IGNORECASE):
                        product_course = c.title()
                        break
                # Also check generic mentions like "the course", "Bootcamp"
                if not product_course:
                    bootcamp_match = re.search(r"\b([A-Za-z0-9+]+(?:\s+[A-Za-z0-9+]+)?\s+(?:course|bootcamp|program|training))\b", text, re.IGNORECASE)
                    if bootcamp_match:
                        product_course = bootcamp_match.group(1).strip().title()

    full_conversation_text = " ".join(combined_user_text)

    # Issue type classification
    lower_text = full_conversation_text.lower()
    if any(k in lower_text for k in ["charged twice", "double charge", "duplicate charge", "charged 2 times"]):
        issue_type = "duplicate_payment"
    elif any(k in lower_text for k in ["refund", "money back", "return money"]):
        issue_type = "refund"
    elif any(k in lower_text for k in ["can't access", "cannot access", "access blocked", "permission denied", "locked out"]):
        issue_type = "course_access"
    elif any(k in lower_text for k in ["video", "playback", "not playing", "black screen", "audio error"]):
        issue_type = "video_error"
    elif any(k in lower_text for k in ["login", "password", "sign in", "auth", "otp"]):
        issue_type = "login"
    elif any(k in lower_text for k in ["assignment", "homework", "quiz", "grading", "mentor"]):
        issue_type = "academic"
    elif full_conversation_text.strip():
        issue_type = "general_support"

    # Description and Summary
    description = full_conversation_text.strip()
    summary = (description[:150] + "...") if len(description) > 150 else description

    extracted = {
        "customer_name": customer_name,
        "contact_info": contact_info,
        "order_or_enrollment_id": order_id,
        "product_or_course": product_course,
        "issue_type": issue_type,
        "description": description,
        "summary": summary,
        "evidence_refs": evidence_refs,
        "session_id": session_id,
    }

    # Verify mandatory fields: DO NOT INFER missing mandatory fields!
    missing = []
    clarification_parts = []
    field_labels = {
        "customer_name": "your name",
        "order_or_enrollment_id": "your enrollment ID or order number (e.g., ORD-12345 or 8-digit ID)",
        "product_or_course": "the specific course or product name",
        "issue_type": "the issue category",
        "description": "a description of the problem",
        "contact_info": "your contact email or phone number",
    }

    for req in mandatory:
        val = extracted.get(req)
        if not val or not str(val).strip():
            missing.append(req)
            label = field_labels.get(req, req.replace("_", " "))
            clarification_parts.append(label)

    if missing:
        if len(clarification_parts) == 1:
            clarification_prompt = f"To create a support ticket, please provide {clarification_parts[0]}."
        else:
            clarification_prompt = (
                f"To create your support ticket, we need a few more details: please share {', '.join(clarification_parts[:-1])} and {clarification_parts[-1]}."
            )
        return ExtractionResult(
            ticket_data=None,
            missing_fields=missing,
            clarification_prompt=clarification_prompt,
            confidence=0.5,
        )

    return ExtractionResult(
        ticket_data=extracted,
        missing_fields=[],
        clarification_prompt=None,
        confidence=1.0,
    )
