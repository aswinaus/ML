import time, json, math, threading, requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
logging.basicConfig(level=logging.INFO)
# Thread-local for a single requests.Session per executor thread
_thread = threading.local()

def _session():
    if not hasattr(_thread, "s"):
        _thread.s = requests.Session()
        _thread.s.headers.update({"User-Agent": "dbx-pipeline/1.0"})
        _thread.s.timeout = 30
    return _thread.s

class RateLimitError(Exception): pass

def is_retryable(resp):
    return resp.status_code in (408, 429, 500, 502, 503, 504)

def raise_for_status_retryable(resp):
    if is_retryable(resp):
        raise RateLimitError(f"retryable status {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()

def rate_limit_sleep(rps: float):
    # simple token-bucket: sleep per request to match ~RPS
    if rps and rps > 0:
        time.sleep(1.0 / rps)

def api_call_decorator(rps: float = 5.0):
    def wrap(func):
        @retry(
            retry=retry_if_exception_type((requests.exceptions.RequestException, RateLimitError)),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(6),
            before_sleep=lambda retry_state: logging.warning(
                f"Retrying {func.__name__} after exception: {retry_state.outcome.exception()}. "
                f"Sleeping for {retry_state.next_action.sleep} seconds."
            ),
        )
        def inner(*args, **kwargs):
            if rps and rps > 0:
                logging.info(f"Throttling: sleeping for {1.0/rps:.2f} seconds before API call.")
            rate_limit_sleep(rps)
            return func(*args, **kwargs)
        return inner
    return wrap

# Example: generic POST with retries & JSON payload or binary content
@api_call_decorator(rps=8.0)
def post_json(url, headers=None, payload=None, content_bytes=None, **kwargs):
    headers = headers or {}
    payload = payload or {}
    s = _session()
    logging.info(f"Making POST request to {url} with payload: {payload} and content_bytes: {bool(content_bytes)}")
    if content_bytes is not None:
        resp = s.post(url, headers=headers, data=content_bytes, **kwargs)
    else:
        resp = s.post(url, headers=headers, json=payload, **kwargs)
    raise_for_status_retryable(resp)
    logging.info(f"POST request successful. Response: {resp.json()}")
    return resp.json()

@api_call_decorator(rps=8.0)
def get_binary(url, headers=None, **kwargs):
    s = _session()
    logging.info(f"Making GET request to {url}")
    resp = s.get(url, headers=headers, **kwargs)
    raise_for_status_retryable(resp)
    logging.info("GET request successful")
    return resp.content