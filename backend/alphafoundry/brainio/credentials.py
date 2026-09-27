"""BRAIN sign-in credentials and the session directory (kept outside the cloud-synced project folder).

Credentials come from environment variables (``BRAIN_EMAIL``/``BRAIN_PASSWORD`` or ``WQB_EMAIL``/``WQB_PASSWORD``)
or from the OS keychain via ``keyring`` (Windows Credential Manager). Nothing secret is written to the project
folder, the database or logs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("alphafoundry.brain")

SERVICE = "AlphaFoundry-BRAIN"
EMAIL_KEY = "__email__"
ENV_PAIRS = (("BRAIN_EMAIL", "BRAIN_PASSWORD"), ("WQB_EMAIL", "WQB_PASSWORD"))


def session_dir() -> Path:
    """Per-user app-data directory (``%LOCALAPPDATA%\\AlphaFoundry``; ``~/.alphafoundry`` elsewhere)."""
    override = os.environ.get("ALPHAFOUNDRY_SESSION_DIR")
    if override:
        p = Path(override)
    elif os.environ.get("LOCALAPPDATA"):
        p = Path(os.environ["LOCALAPPDATA"]) / "AlphaFoundry"
    else:
        p = Path.home() / ".alphafoundry"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _keyring() -> Any | None:
    try:
        import keyring
        from keyring.backends import fail
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    try:
        kr = keyring.get_keyring()
    except Exception:  # noqa: BLE001
        return None
    if isinstance(kr, fail.Keyring) or getattr(kr, "priority", 1) <= 0:
        return None
    return keyring


def keyring_available() -> bool:
    return _keyring() is not None


def env_credentials() -> tuple[str, str] | None:
    for ek, pk in ENV_PAIRS:
        email, pw = os.environ.get(ek, "").strip(), os.environ.get(pk, "")
        if email and pw:
            return email, pw
    return None


def saved_email() -> str | None:
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(SERVICE, EMAIL_KEY) or None
    except Exception:  # noqa: BLE001
        return None


def credentials_saved() -> bool:
    return env_credentials() is not None or _keyring_credentials() is not None


def _keyring_credentials() -> tuple[str, str] | None:
    kr = _keyring()
    if kr is None:
        return None
    try:
        email = kr.get_password(SERVICE, EMAIL_KEY)
        pw = kr.get_password(SERVICE, email) if email else None
    except Exception:  # noqa: BLE001
        log.warning("could not read BRAIN credentials from the OS keychain")
        return None
    return (email, pw) if email and pw else None


def load_credentials() -> tuple[str, str] | None:
    """(email, password) from the environment, then the OS keychain; None when nothing is stored."""
    return env_credentials() or _keyring_credentials()


def save_credentials(email: str, password: str) -> bool:
    """Store credentials in the OS keychain. Returns False when no keychain backend is available."""
    kr = _keyring()
    if kr is None:
        return False
    try:
        old = kr.get_password(SERVICE, EMAIL_KEY)
        if old and old != email:
            try:
                kr.delete_password(SERVICE, old)
            except Exception:  # noqa: BLE001
                pass
        kr.set_password(SERVICE, EMAIL_KEY, email)
        kr.set_password(SERVICE, email, password)
        return True
    except Exception:  # noqa: BLE001
        log.warning("could not save BRAIN credentials to the OS keychain")
        return False


def forget_credentials() -> bool:
    kr = _keyring()
    if kr is None:
        return False
    ok = True
    try:
        email = kr.get_password(SERVICE, EMAIL_KEY)
    except Exception:  # noqa: BLE001
        return False
    for key in ([email] if email else []) + [EMAIL_KEY]:
        try:
            kr.delete_password(SERVICE, key)
        except Exception:  # noqa: BLE001
            ok = False
    return ok


__all__ = ["session_dir", "load_credentials", "save_credentials", "forget_credentials", "keyring_available",
           "credentials_saved", "saved_email", "env_credentials"]
