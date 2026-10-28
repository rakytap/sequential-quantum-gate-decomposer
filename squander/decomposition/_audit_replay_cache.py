"""Scoped, byte-bounded exact-content memoization for independent audit passes."""

from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
import json
import sys
import time

_CURRENT = ContextVar("squander_audit_replay_cache", default=None)


class ExactJSONCache:
    """Cache full keys, not hashes; eviction only causes recomputation."""

    def __init__(self, max_bytes=32 * 1024 * 1024, max_entries=32768):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key, compute):
        encoded = self.entries.get(key)
        if encoded is not None:
            self.hits += 1
            self.entries.move_to_end(key)
            # Every caller receives a fresh value, never shared mutable state.
            return json.loads(encoded)
        self.misses += 1
        value = compute()
        encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        weight = sys.getsizeof(key) + sys.getsizeof(encoded) + 128
        if (self.max_entries > 0 and weight <= self.max_bytes // 4):
            while self.entries and (
                self.bytes + weight > self.max_bytes
                or len(self.entries) >= self.max_entries
            ):
                old_key, old_value = self.entries.popitem(last=False)
                self.bytes -= sys.getsizeof(old_key) + sys.getsizeof(old_value) + 128
            if self.bytes + weight <= self.max_bytes:
                self.entries[key] = encoded
                self.bytes += weight
        return value


class AuditReplayCache:
    def __init__(self):
        self.states = ExactJSONCache()
        self.metrics = ExactJSONCache()

    def __enter__(self):
        self.token = _CURRENT.set(self)
        return self

    def __exit__(self, *exc):
        _CURRENT.reset(self.token)


def current_cache():
    return _CURRENT.get()


def independent_audit_pass(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        # In particular, export and verification never share numerical results.
        with AuditReplayCache():
            return function(*args, **kwargs)
    return wrapped


class AuditProgress:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.started = time.monotonic()
        self.last_report = self.started
        self.phase = None

    def tick(self, phase, completed=0, total=None, force=False):
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and phase == self.phase and now - self.last_report < 10:
            return
        self.phase = phase
        self.last_report = now
        count = str(completed) if total is None else f"{completed}/{total}"
        print(f"Rewrite audit {phase}: {count}; {now-self.started:.2f} s.", flush=True)

