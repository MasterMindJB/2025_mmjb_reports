"""Retry a Gemini call when the service times out or returns a 5xx.

Validation failures are not retried here. The story loop does that itself,
and it keeps the reason.
"""

from __future__ import annotations

import logging
import time

RETRYABLE_NAMES = {
    'DeadlineExceeded',
    'ServiceUnavailable',
    'InternalServerError',
    'GatewayTimeout',
    'TooManyRequests',
    'ResourceExhausted',
}


def retryable_gemini_error(exc: BaseException) -> bool:
    if type(exc).__name__ in RETRYABLE_NAMES:
        return True
    code = getattr(exc, 'code', None)
    if isinstance(code, int) and (code == 429 or code >= 500):
        return True
    text = str(exc)
    return any(token in text for token in ('504', '503', '502', '500', '429', 'Deadline'))


def invoke_llm(llm, prompt, *, attempts: int = 4, pause=time.sleep):
    """Call llm.invoke, backing off on 504, DeadlineExceeded, and other 5xx."""
    delay = 1.0
    last = None
    for attempt in range(attempts):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            last = exc
            if not retryable_gemini_error(exc) or attempt + 1 == attempts:
                raise
            logging.warning('gemini retry after %s', exc)
            pause(delay)
            delay = min(delay * 2, 8)
    raise last
