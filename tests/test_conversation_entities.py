"""Tests for entity extraction, preservation, and restoration in conversations."""

from support_bot.conversation.entities import (
    protect_entities,
    restore_entities,
)


def test_entity_preservation_order_and_payment_ids():
    text = "My order is ORD-4521 and payment reference is PAY_998812."
    protected = protect_entities(text)

    # Check placeholders replaced the real IDs
    assert "ORD-4521" not in protected.text
    assert "PAY_998812" not in protected.text
    assert "__ENT_ID_" in protected.text or "__ENT_CONTEXTUAL_ID_" in protected.text

    # Simulate translation that could alter surround words
    simulated_translation = f"Mera kramank {list(protected.entity_map.keys())[0]} aur bhugtan {list(protected.entity_map.keys())[1]} hai."

    # Restore must recover the exact original IDs
    restored = protected.restore(simulated_translation)
    assert "ORD-4521" in restored
    assert "PAY_998812" in restored


def test_entity_preservation_product_codes_currency_dates():
    text = "Enrolled in course PY-101 on 2025-01-06 for ₹5000 with course DS-202."
    protected = protect_entities(text)

    # All critical entities should be in entity map
    values = list(protected.entity_map.values())
    assert "PY-101" in values
    assert "DS-202" in values
    assert "₹5000" in values
    assert "2025-01-06" in values

    # Simulate translation to Hindi
    simulated_hindi = protected.text.replace("Enrolled in course", "कोर्स में दाखिला लिया").replace("for", "के लिए")

    # Restore verifies they are never altered
    restored = protected.restore(simulated_hindi)
    assert "PY-101" in restored
    assert "DS-202" in restored
    assert "₹5000" in restored
    assert "2025-01-06" in restored


def test_contextual_order_id_patterns():
    samples = [
        "Please check order 4521 status",
        "Issue with enrollment id ENR-9901",
        "reference # REF123456",
    ]
    for s in samples:
        protected = protect_entities(s)
        assert len(protected.entity_map) >= 1
        # Proves round-trip restoration
        assert protected.restore(protected.text) == s


def test_extra_entities_from_facts():
    text = "Hi, this is Rohan Sharma asking about refund."
    extra = {"customer_name": "Rohan Sharma"}
    protected = protect_entities(text, extra_entities=extra)

    assert "Rohan Sharma" not in protected.text
    assert "__ENT_CUSTOMER_NAME_" in protected.text
    restored = protected.restore(f"Namaste, {list(protected.entity_map.keys())[0]} refund ke bare me pooch rahe hain.")
    assert "Rohan Sharma" in restored
