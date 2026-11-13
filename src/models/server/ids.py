"""Random identifiers of the API's objects."""

import secrets
import string

ALNUM = string.ascii_letters + string.digits


def hex_id(prefix: str, nbytes: int) -> str:
    """``prefix`` and ``nbytes`` random bytes in hex (``chatcmpl-``, ``item_``, ...)."""
    return prefix + secrets.token_hex(nbytes)


def alnum_id(n: int, prefix: str = "") -> str:
    """``prefix`` and ``n`` random letters and digits."""
    return prefix + "".join(secrets.choice(ALNUM) for _ in range(n))
