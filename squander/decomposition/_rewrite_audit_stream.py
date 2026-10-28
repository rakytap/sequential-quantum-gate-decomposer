"""Bounded-memory storage for schema-v2 rewrite certificates.

Small certificates retain their ordinary mutable list API. Large certificates
use a disposable SQLite index beside the archive, never in the small /tmp
filesystem. This changes storage only, not certificate contents or checks.
"""

from collections.abc import Sequence
import gzip
import heapq
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from ._audit_replay_cache import AuditProgress

MAX_MEMORY_BYTES = 8 * 1024 * 1024
CHUNK_CHARS = 64 * 1024


class DiskAuditEvents(Sequence):
    """Read-only, repeatable event stream with indexed event-ID lookups."""

    def __init__(self, directory):
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".rewrite-audit-index-", dir=directory)
        self.connection = sqlite3.connect(
            os.path.join(self.temporary.name, "events.sqlite"))
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-8192")
        self.connection.execute(
            "CREATE TABLE events (seq INTEGER PRIMARY KEY, event_id TEXT UNIQUE, "
            "kind TEXT, timestamp_ns INTEGER, started_ns INTEGER, payload TEXT)")
        self.count = 0

    def add(self, event, encoded):
        event_id = event.get("event_id")
        if event_id is not None and not isinstance(event_id, str):
            raise ValueError("Rewrite event IDs must be strings.")
        try:
            self.connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (self.count, event_id, event.get("kind"),
                 int(event.get("timestamp_ns", 0)),
                 int(event.get("started_ns", 0)), encoded))
        except sqlite3.IntegrityError as error:
            raise AssertionError(
                f"Duplicate rewrite audit event ID {event_id!r}.") from error
        self.count += 1

    def finish(self):
        self.connection.execute(
            "CREATE INDEX event_order ON events(timestamp_ns, event_id, seq)")
        self.connection.execute(
            "CREATE INDEX kind_order ON events(kind, started_ns, timestamp_ns, event_id, seq)")
        self.connection.execute(
            "CREATE INDEX kind_timestamp_order ON events(kind, timestamp_ns, event_id, seq)")
        self.connection.execute(
            "CREATE INDEX started_order ON events(started_ns, timestamp_ns, event_id, seq)")
        self.connection.commit()
        return self

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.count))]
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        row = self.connection.execute(
            "SELECT payload FROM events ORDER BY timestamp_ns, event_id, seq "
            "LIMIT 1 OFFSET ?", (index,)).fetchone()
        return json.loads(row[0])

    def get(self, event_id, default=None):
        row = self.connection.execute(
            "SELECT payload FROM events WHERE event_id=?", (event_id,)).fetchone()
        return default if row is None else json.loads(row[0])

    def raw_events(self, kinds=None, ordered=False):
        order = ("started_ns, timestamp_ns, event_id, seq"
                 if ordered else "timestamp_ns, event_id, seq")
        if kinds is None:
            index = "started_order" if ordered else "event_order"
            cursor = self.connection.execute(
                "SELECT payload FROM events INDEXED BY " + index + " ORDER BY " + order)
            try:
                for row in cursor:
                    yield row[0]
            finally:
                cursor.close()
            return
        # Merge ordered per-kind cursor heads, not whole large payloads.
        index = "kind_order" if ordered else "kind_timestamp_order"
        fields = ("started_ns, timestamp_ns, event_id, seq"
                  if ordered else "timestamp_ns, event_id, seq")
        cursors = []
        try:
            for kind in sorted(set(kinds)):
                cursors.append(self.connection.execute(
                    "SELECT payload, " + fields + " FROM events INDEXED BY "
                    + index + " WHERE kind=? ORDER BY " + order, (kind,)))
            def key(row):
                values = list(row[1:])
                values[-2] = (values[-2] is not None, values[-2] or "")
                return tuple(values)
            for row in heapq.merge(*cursors, key=key):
                yield row[0]
        finally:
            for cursor in cursors:
                cursor.close()

    def iter_events(self, kinds=None, ordered=False):
        for encoded in self.raw_events(kinds, ordered):
            yield json.loads(encoded)

    def __iter__(self):
        return self.iter_events()

    def close(self):
        connection = getattr(self, "connection", None)
        if connection is not None:
            self.connection = None
            connection.close()
        temporary = getattr(self, "temporary", None)
        if temporary is not None:
            self.temporary = None
            temporary.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _EventBuilder:
    def __init__(self, directory):
        self.directory = directory
        self.events = []
        self.disk = None
        self.bytes = 0

    def add(self, event):
        if not isinstance(event, dict):
            raise ValueError("Rewrite audit events must be JSON objects.")
        encoded = json.dumps(event, separators=(",", ":"), allow_nan=False)
        self.bytes += len(encoded.encode("utf-8"))
        if self.disk is None and self.bytes > MAX_MEMORY_BYTES:
            self.disk = DiskAuditEvents(self.directory)
            for previous in self.events:
                self.disk.add(previous, json.dumps(
                    previous, separators=(",", ":"), allow_nan=False))
            self.events.clear()
        if self.disk is None:
            self.events.append(event)
        else:
            self.disk.add(event, encoded)

    def finish(self):
        if self.disk is None:
            self.events.sort(key=lambda event: (
                event.get("timestamp_ns", 0), event.get("event_id", "")))
            result, self.events = self.events, []
            return result
        result = self.disk.finish()
        self.disk = None
        return result

    def close(self):
        if self.disk is not None:
            self.disk.close()
            self.disk = None


class _JSONReader:
    """Incrementally decode one JSON value at a time using the stdlib parser."""

    def __init__(self, source):
        self.source = source
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def fill(self):
        tail = self.buffer[self.position:]
        chunk = self.source.read(CHUNK_CHARS)
        self.buffer = tail + chunk
        self.position = 0
        self.eof = not chunk

    def peek(self):
        while True:
            while (self.position < len(self.buffer)
                   and self.buffer[self.position].isspace()):
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                return ""
            self.fill()

    def expect(self, token):
        if self.peek() != token:
            raise ValueError(f"Malformed rewrite audit: expected {token!r}.")
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.fill()
                continue
            # A number can continue in the next chunk; do not accept a prefix.
            continuing_number = (
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and end < len(self.buffer) and self.buffer[end] in ".eE"
            )
            if not self.eof and (end == len(self.buffer) or continuing_number):
                self.fill()
                continue
            self.position = end
            return value


def read_archive(source, add_event):
    """Read the envelope without ever decoding the events array as a list."""
    reader = _JSONReader(source)
    reader.expect("{")
    header = {}
    saw_events = False
    if reader.peek() != "}":
        while True:
            key = reader.value()
            if not isinstance(key, str) or key in header or (key == "events" and saw_events):
                raise ValueError("Malformed or duplicate rewrite audit key.")
            reader.expect(":")
            if key == "events":
                saw_events = True
                reader.expect("[")
                if reader.peek() != "]":
                    while True:
                        add_event(reader.value())
                        if reader.peek() == "]":
                            break
                        reader.expect(",")
                reader.expect("]")
            else:
                header[key] = reader.value()
            if reader.peek() == "}":
                break
            reader.expect(",")
    reader.expect("}")
    if reader.peek() or not saw_events:
        raise ValueError("Malformed rewrite audit envelope.")
    return header


def load_audit(path):
    builder = _EventBuilder(Path(path).resolve().parent)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            audit = read_archive(source, builder.add)
        audit["events"] = builder.finish()
        return audit
    finally:
        builder.close()


def close_audit(audit):
    events = audit.get("events")
    if isinstance(events, DiskAuditEvents):
        events.close()


def iter_audit_events(events, kinds=None, ordered=False):
    if isinstance(events, DiskAuditEvents):
        return events.iter_events(kinds, ordered)
    selected = (event for event in events
                if kinds is None or event.get("kind") in kinds)
    return iter(sorted(selected, key=lambda event: event["started_ns"])) if ordered else selected


def save_audit(jsonl_path, output_path, run, schema_version, prepare):
    """Compute every original metric, externally sort, and atomically export."""
    builder = _EventBuilder(Path(output_path).resolve().parent)
    progress = AuditProgress(run.get("audit_progress", False))
    progress.tick("export proofs", force=True)
    summary = dict(events=0, rewrites=0, accepted_rewrites=0,
                   failed_synthesis_attempts=0, fallbacks=0,
                   candidate_tolerance_misses=0)
    events = None
    temporary_path = os.fspath(output_path) + ".tmp"
    try:
        if os.path.exists(jsonl_path):
            with open(jsonl_path, encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise AssertionError(
                            f"Corrupt audit event at {jsonl_path}:{line_number}.") from error
                    prepare(event)
                    rewrite = event.get("kind") == "rewrite"
                    summary["events"] += 1
                    summary["rewrites"] += rewrite
                    summary["accepted_rewrites"] += rewrite and bool(event.get("accepted", False))
                    summary["failed_synthesis_attempts"] += (
                        event.get("kind") == "synthesis_attempt"
                        and not event.get("accepted", False))
                    summary["fallbacks"] += bool(event.get("fallback_used", False))
                    summary["candidate_tolerance_misses"] += (
                        rewrite and not event.get("within_tolerance", True)
                        and not event.get("accepted", False))
                    builder.add(event)
                    if summary["events"] % 10000 == 0:
                        progress.tick("export proofs", summary["events"])
        progress.tick("index proofs", summary["events"], force=True)
        events = builder.finish()
        progress.tick("write archive", 0, len(events), force=True)
        audit = dict(schema_version=schema_version, run=run, summary=summary)
        with gzip.open(temporary_path, "wt", encoding="utf-8") as output:
            output.write(json.dumps(audit, separators=(",", ":"), allow_nan=False)[:-1])
            output.write(',"events":[')
            raw_events = (events.raw_events() if isinstance(events, DiskAuditEvents)
                          else (json.dumps(event, separators=(",", ":"), allow_nan=False)
                                for event in events))
            for index, encoded in enumerate(raw_events):
                if index % 10000 == 0:
                    progress.tick("write archive", index, len(events))
                if index:
                    output.write(",")
                output.write(encoded)
            output.write("]}")
        os.replace(temporary_path, output_path)
        progress.tick("archive complete", len(events), len(events), force=True)
        audit["events"] = events
        return audit
    except BaseException:
        if isinstance(events, DiskAuditEvents):
            events.close()
        raise
    finally:
        builder.close()
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
