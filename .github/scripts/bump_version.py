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

    version_file_paths = [
        Path("setup.py"),
        Path("CMakeLists.txt"),
        Path("Doxyfile"),
        Path("Doxyfile"),
    ]

    search_patterns = [
        r'version\s*=\s*["\']([^"\']+)["\']',
        r"CQGD VERSION\s+(\d+\.\d+\.\d+)",
        r"PROJECT_NUMBER\s*=\s*v?(\d+\.\d+\.\d+)",
        r"version=v?(\d+\.\d+\.\d+)",
    ]
    print(search_patterns)

    for version_file_path, search_pattern in zip(version_file_paths, search_patterns):

        if not version_file_path.exists():
            raise SystemExit(
                f"{version_file_path} not found; version bump script needs it."
            )

        text = version_file_path.read_text(encoding="utf-8")
        match = re.search(search_pattern, text)
        if not match:
            raise SystemExit(
                f"Could not locate a version declaration inside {version_file_path}"
            )

        current_version = Version(match.group(1))

        if segment == "major":
            new_version = Version(f"{current_version.major + 1}.0.0")
        elif segment == "minor":
            new_version = Version(
                f"{current_version.major}.{current_version.minor + 1}.0"
            )
        else:
            new_version = Version(
                f"{current_version.major}.{current_version.minor}.{current_version.micro + 1}"
            )

        updated_text = text[: match.start(1)] + str(new_version) + text[match.end(1) :]
        version_file_path.write_text(updated_text, encoding="utf-8")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"version={new_version}\n")

    print(f"Bumped version from {current_version} to {new_version}")


if __name__ == "__main__":
    main()
