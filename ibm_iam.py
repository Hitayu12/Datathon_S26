"""IBM IAM access token helper with simple module-level caching."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import requests

IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"
_CACHED_TOKEN: Optional[str] = None
_CACHED_EXPIRES_AT: float = 0.0


def get_iam_token(api_key: str) -> Tuple[str, float]:
    """Exchange an IBM Cloud API key for an IAM bearer token.

    Returns:
        A tuple of (access_token, expires_at_epoch_seconds).

    Raises:
        ValueError: If the API key is empty.
        RuntimeError: If the IAM request fails or returns an invalid payload.
    """

    global _CACHED_TOKEN, _CACHED_EXPIRES_AT

    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise ValueError("IBM IAM API key is required.")

    now = time.time()
    if _CACHED_TOKEN and (_CACHED_EXPIRES_AT - now) > 60:
        return _CACHED_TOKEN, _CACHED_EXPIRES_AT

    payload = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": cleaned_key,
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }

    try:
        response = requests.post(
            IAM_TOKEN_URL,
            data=payload,
            headers=headers,
            timeout=20,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"IBM IAM token request failed: {exc}") from exc

    if not response.ok:
        detail = response.text.strip() or f"HTTP {response.status_code}"
        raise RuntimeError(
            f"IBM IAM token request was rejected ({response.status_code}): {detail}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("IBM IAM token response was not valid JSON.") from exc

    access_token = body.get("access_token")
    expiration = body.get("expiration")
    expires_in = body.get("expires_in")

    if not isinstance(access_token, str) or not access_token.strip():
        raise RuntimeError("IBM IAM token response did not include a valid access_token.")

    expires_at: Optional[float] = None
    if isinstance(expiration, (int, float)):
        expires_at = float(expiration)
    elif isinstance(expires_in, (int, float)):
        expires_at = now + float(expires_in)

    if expires_at is None or expires_at <= now:
        raise RuntimeError("IBM IAM token response did not include a valid expiration time.")

    _CACHED_TOKEN = access_token
    _CACHED_EXPIRES_AT = expires_at
    return _CACHED_TOKEN, _CACHED_EXPIRES_AT
