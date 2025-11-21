import os
import re
from pathlib import Path

from packaging.version import Version


def main() -> None:
    segment = os.environ.get("BUMP_SEGMENT", "patch").lower()
    valid_segments = {"major", "minor", "patch"}
    if segment not in valid_segments:
        raise SystemExit(
            f"Unsupported bump segment: {segment} (expected one of {sorted(valid_segments)})"
        )

    setup_path = Path("setup.py")
    if not setup_path.exists():
        raise SystemExit("setup.py not found; version bump script needs it.")

    text = setup_path.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        raise SystemExit("Could not locate a version declaration inside setup.py")

    current_version = Version(match.group(1))

    if segment == "major":
        new_version = Version(f"{current_version.major + 1}.0.0")
    elif segment == "minor":
        new_version = Version(f"{current_version.major}.{current_version.minor + 1}.0")
    else:
        new_version = Version(
            f"{current_version.major}.{current_version.minor}.{current_version.micro + 1}"
        )

    updated_text = text[: match.start(1)] + str(new_version) + text[match.end(1) :]
    setup_path.write_text(updated_text, encoding="utf-8")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"version={new_version}\n")

    print(f"Bumped version from {current_version} to {new_version}")


if __name__ == "__main__":
    main()

