"""Read-only HTTP readers for historical Kalshi and Polymarket data.

Both readers issue GET requests only; there is no code path that can place,
change or cancel an order. Kalshi requests are signed when an API key is
configured, which raises the rate limit from the anonymous per-IP budget to the
account's read budget; the key never leaves this process.

Credentials follow the repository convention: ``KALSHI_API_KEY_ID`` and
``KALSHI_PRIVATE_KEY_PATH`` from the environment, or the same two names read
from an env file (a relative key path is resolved against the env file's
directory).
"""

from __future__ import annotations

import base64
import os
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx

KALSHI_REST = 'https://external-api.kalshi.com/trade-api/v2'
GAMMA_API = 'https://gamma-api.polymarket.com'
POLY_DATA_API = 'https://data-api.polymarket.com'
USER_AGENT = (
    'oracle3-ml (+https://github.com/YichengYang-Ethan/oracle3-prediction-market-agent)'
)


class SourceError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f'{status} from {url}: {body[:200]}')
        self.status = status


class RateLimiter:
    """At most ``rate`` requests per second across all threads, evenly spaced."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


def read_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        out[key.replace('export ', '').strip()] = value.strip().strip('"').strip("'")
    return out


class _KalshiSigner:
    def __init__(self, key_id: str, pem_path: Path):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        self.key_id = key_id
        key = serialization.load_pem_private_key(pem_path.read_bytes(), password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError(
                f'{pem_path} is not an RSA private key; Kalshi keys are RSA'
            )
        self._key = key

    def headers(self, path: str) -> dict[str, str]:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        ts = str(int(time.time() * 1000))
        sig = self._key.sign(
            (ts + 'GET' + path).encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256(),
        )
        return {
            'KALSHI-ACCESS-KEY': self.key_id,
            'KALSHI-ACCESS-TIMESTAMP': ts,
            'KALSHI-ACCESS-SIGNATURE': base64.b64encode(sig).decode(),
        }


def kalshi_signer_from_env(env_file: str | None = None) -> _KalshiSigner | None:
    """Signer from ``env_file`` or the process environment; None when no key is configured."""
    values: Mapping[str, str] = read_env_file(env_file) if env_file else os.environ
    key_id = values.get('KALSHI_API_KEY_ID')
    pem = values.get('KALSHI_PRIVATE_KEY_PATH')
    if not key_id or not pem:
        return None
    pem_path = Path(pem).expanduser()
    if not pem_path.is_absolute() and env_file:
        pem_path = Path(env_file).expanduser().parent / pem_path
    return _KalshiSigner(key_id, pem_path)


class _Reader:
    """Thread-safe GET-only JSON reader with pacing and retries."""

    def __init__(self, base: str, rate: float, retries: int = 6, timeout: float = 30.0):
        self.base = base.rstrip('/')
        self.limiter = RateLimiter(rate)
        self.retries = retries
        self.timeout = timeout
        self._local = threading.local()
        self.stats = {'ok': 0, 'retry': 0, 'error': 0}

    def _client(self) -> httpx.Client:
        client = getattr(self._local, 'client', None)
        if client is None:
            client = httpx.Client(
                timeout=self.timeout,
                headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'},
            )
            self._local.client = client
        return client

    def _headers(self, path: str) -> dict[str, str]:
        return {}

    def get(self, path: str, **params: Any) -> Any:
        url = self.base + path
        clean = {k: v for k, v in params.items() if v is not None}
        last: Exception | None = None
        for attempt in range(self.retries):
            self.limiter.wait()
            try:
                resp = self._client().get(
                    url, params=clean, headers=self._headers(httpx.URL(url).path)
                )
            except httpx.HTTPError as exc:
                last = exc
                self.stats['retry'] += 1
                time.sleep(min(30.0, 0.5 * 2**attempt))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last = SourceError(resp.status_code, url, resp.text)
                self.stats['retry'] += 1
                time.sleep(min(30.0, 0.5 * 2**attempt))
                continue
            if resp.status_code != 200:
                self.stats['error'] += 1
                raise SourceError(resp.status_code, url, resp.text)
            self.stats['ok'] += 1
            return resp.json()
        self.stats['error'] += 1
        raise last if last else RuntimeError(f'request failed: {url}')


class KalshiReader(_Reader):
    def __init__(self, signer: _KalshiSigner | None = None, rate: float | None = None):
        # Signed Basic tier: 200 read tokens/s = 20 requests/s; anonymous: about 5/s per IP.
        super().__init__(KALSHI_REST, rate or (18.0 if signer else 4.0), retries=10)
        self.signer = signer

    def _headers(self, path: str) -> dict[str, str]:
        return self.signer.headers(path) if self.signer else {}

    def paginate(
        self, path: str, key: str, max_pages: int = 10_000, **params: Any
    ) -> Iterator[list[dict]]:
        cursor = None
        for _ in range(max_pages):
            page = self.get(path, cursor=cursor, **params)
            yield page.get(key) or []
            cursor = page.get('cursor')
            if not cursor:
                return


class GammaReader(_Reader):
    def __init__(self, rate: float = 3.0):
        super().__init__(GAMMA_API, rate, retries=8)


class PolyDataReader(_Reader):
    def __init__(self, rate: float = 14.0):
        super().__init__(POLY_DATA_API, rate)
