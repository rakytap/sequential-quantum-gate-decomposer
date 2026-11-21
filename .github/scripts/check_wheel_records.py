"""Verify wheel file coverage, hashes, and sizes without installing or editing it."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import glob
import hashlib
import io
from pathlib import Path
import sys
import zipfile


class WheelRecordError(ValueError):
    """The wheel's RECORD does not describe its archived files."""


def check_wheel(path: Path) -> int:
    """Return the number of hashed files, or raise on any inconsistent entry."""
    with zipfile.ZipFile(path) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        if duplicates:
            raise WheelRecordError(f"duplicate archive member: {duplicates[0]}")
        records = [name for name in names
                   if name.count("/") == 1 and name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise WheelRecordError("expected exactly one root .dist-info/RECORD")
        record = records[0]
        rows = {}
        with archive.open(record) as source:
            for row in csv.reader(io.TextIOWrapper(source, encoding="utf-8", newline=""),
                                  strict=True):
                if len(row) != 3:
                    raise WheelRecordError(f"invalid RECORD row: {row!r}")
                name, digest, size = row
                if name in rows:
                    raise WheelRecordError(f"duplicate RECORD entry: {name}")
                rows[name] = (digest, size)

        # Legacy signatures are the only other files exempt from RECORD.
        signatures = {record + ".jws", record + ".p7s"}
        files = set(names)
        missing = (files - signatures) - rows.keys()
        extra = rows.keys() - files
        if missing:
            raise WheelRecordError(f"file missing from RECORD: {sorted(missing)[0]}")
        if extra:
            raise WheelRecordError(f"RECORD lists absent file: {sorted(extra)[0]}")
        if signatures & rows.keys():
            raise WheelRecordError("legacy signature files must not be listed in RECORD")
        if rows[record] != ("", ""):
            raise WheelRecordError("RECORD's own hash and size must be empty")

        for name, (expected_hash, expected_size) in rows.items():
            if name == record:
                continue
            algorithm, separator, encoded = expected_hash.partition("=")
            if not separator or not encoded:
                raise WheelRecordError(f"missing or invalid hash: {name}")
            try:
                hasher = hashlib.new(algorithm)
            except ValueError as error:
                raise WheelRecordError(f"unsupported hash for {name}: {algorithm}") from error
            if hasher.digest_size < 32:
                raise WheelRecordError(f"hash must be SHA-256 or stronger: {name}")
            try:
                recorded_size = int(expected_size)
            except ValueError as error:
                raise WheelRecordError(f"missing or invalid size: {name}") from error
            actual_size = 0
            with archive.open(name) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(chunk)
                    actual_size += len(chunk)
            actual_hash = base64.urlsafe_b64encode(hasher.digest()).rstrip(b"=").decode("ascii")
            if actual_hash != encoded:
                raise WheelRecordError(f"hash mismatch: {name}")
            if actual_size != recorded_size:
                raise WheelRecordError(f"size mismatch: {name} ({recorded_size} != {actual_size})")
        return len(rows) - 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", help="Wheel paths or glob patterns")
    args = parser.parse_args(argv)
    paths = set()
    for pattern in args.wheels:
        matches = glob.glob(pattern)
        if not matches:
            parser.error(f"no wheels match {pattern!r}")
        paths.update(Path(name) for name in matches)
    failed = False
    for path in sorted(paths):
        try:
            count = check_wheel(path)
        except (WheelRecordError, OSError, zipfile.BadZipFile, UnicodeError, csv.Error) as error:
            print(f"FAIL {path}: {error}", file=sys.stderr)
            failed = True
        else:
            print(f"PASS {path}: {count} file hashes and sizes verified")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
