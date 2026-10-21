"""Tests for support_bot.masking — PII and payment data masking and logging filter."""

import logging
from support_bot.masking import MaskingFilter, mask_pii


def test_mask_email():
    text = "Please reach out to support@example.com or user.name+tag@sub.domain.co.in"
    masked = mask_pii(text)
    assert "[EMAIL]" in masked
    assert "support@example.com" not in masked
    assert "user.name+tag@sub.domain.co.in" not in masked


def test_mask_phone():
    samples = [
        "Call me at +91 9876543210 today.",
        "Contact: 09876543210 immediately.",
        "Phone is 98765 43210 thanks.",
        "Direct: 9876543210.",
    ]
    for sample in samples:
        masked = mask_pii(sample)
        assert "[PHONE]" in masked
        assert "9876543210" not in masked
        assert "98765 43210" not in masked


def test_mask_card():
    samples = [
        "Card: 4111 2222 3333 4444 expires 12/28",
        "Payment via 4111-2222-3333-4444",
        "Raw digits 4111222233334444 charged",
    ]
    for sample in samples:
        masked = mask_pii(sample)
        assert "[CARD]" in masked
        assert "4111" not in masked


def test_mask_bank_account_and_ifsc():
    text = "Account: 123456789012345 at IFSC HDFC0001234"
    masked = mask_pii(text)
    assert "[BANK_ACCT]" in masked
    assert "123456789012345" not in masked
    assert "[IFSC]" in masked
    assert "HDFC0001234" not in masked


def test_mask_upi():
    samples = [
        "Send money to user@upi",
        "Pay to merchant@okaxis",
        "UPI: test.user@paytm",
    ]
    for sample in samples:
        masked = mask_pii(sample)
        assert "[UPI]" in masked


def test_mask_api_key():
    samples = [
        "Secret: AIzaSyDfakekey1234567890ABCDEF",
        "Key: sk-fakeapikey1234567890123456",
        "Github: ghp_faketoken1234567890123456",
    ]
    for sample in samples:
        masked = mask_pii(sample)
        assert "[API_KEY]" in masked
        assert "fakekey" not in masked


def test_mask_plain_text():
    text = "What is the return policy for defective laptops?"
    assert mask_pii(text) == text
    assert mask_pii("") == ""


def test_masking_filter():
    logger = logging.getLogger("test_masking_filter")
    logger.setLevel(logging.INFO)
    records = []

    class CapturingHandler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = CapturingHandler()
    handler.addFilter(MaskingFilter())
    logger.addHandler(handler)

    try:
        logger.info("User email is john.doe@example.com with phone 9876543210")
        assert len(records) == 1
        assert "john.doe@example.com" not in records[0]
        assert "9876543210" not in records[0]
        assert "[EMAIL]" in records[0]
        assert "[PHONE]" in records[0]
    finally:
        logger.removeHandler(handler)
