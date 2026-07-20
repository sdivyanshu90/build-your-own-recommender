"""Pseudonymous identifier logging helper."""

import hashlib
import hmac


def hash_identifier(identifier: str, salt: str) -> str:
    if len(salt) < 16:
        raise ValueError("identifier hash salt must contain at least 16 characters")
    return hmac.new(salt.encode(), identifier.encode(), hashlib.sha256).hexdigest()[:20]
