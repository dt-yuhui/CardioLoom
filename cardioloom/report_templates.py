"""Privacy-preserving report-template normalization for internal training."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_report_template(value: str) -> str:
    """Return the exact normalization used by the report-corpus audit."""

    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def stable_report_template_id(normalized_template: str) -> int:
    """Map a non-empty template to a stable signed int64 identifier.

    The identifier is used only inside the data pipeline and DDP positive mask;
    it is never exported as a corpus artifact.
    """

    if not normalized_template:
        raise ValueError("Cannot assign an ID to an empty report template")
    digest = hashlib.blake2b(
        normalized_template.encode("utf-8"),
        digest_size=8,
        person=b"ECGCLIP-RPT",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=True)
