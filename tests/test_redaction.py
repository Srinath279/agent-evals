from agent_evals.core.redaction import redact, redact_text


def test_redacts_email_phone_ssn_card_ip():
    text = (
        "Contact jane@example.com or 415-555-0142. SSN 123-45-6789, "
        "card 4111 1111 1111 1111, server 192.168.0.1."
    )
    out = redact_text(text)
    for pii in ["jane@example.com", "123-45-6789", "4111 1111 1111 1111", "192.168.0.1"]:
        assert pii not in out
    for token in ["<EMAIL>", "<PHONE>", "<SSN>", "<CARD_NUMBER>", "<IP_ADDRESS>"]:
        assert token in out


def test_redact_recurses_structures():
    obj = {"customer": {"email": "a@b.co"}, "notes": ["call 415-555-0142", 42]}
    out = redact(obj)
    assert out["customer"]["email"] == "<EMAIL>"
    assert "<PHONE>" in out["notes"][0]
    assert out["notes"][1] == 42
