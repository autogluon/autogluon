import json
import logging
import warnings
from collections import Counter
from subprocess import PIPE, Popen


def test_check_style():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Ruff style check")

    ruff_proc = Popen(
        ["ruff", "check", ".", "--exit-zero", "--output-format", "json"],
        stdout=PIPE,
        stderr=PIPE,
    )

    stdout, stderr = ruff_proc.communicate()
    stdout_str = stdout.decode()
    stderr_str = stderr.decode()

    if stderr_str.strip():
        print("\n=== Ruff stderr ===")
        print(stderr_str)

    try:
        diagnostics = json.loads(stdout_str) if stdout_str.strip() else []
    except json.JSONDecodeError as e:
        print("\n=== Ruff stdout (failed to parse as JSON) ===")
        print(stdout_str)
        raise AssertionError(f"Failed to parse Ruff JSON output: {e}") from e

    total_count = len(diagnostics)

    if total_count > 0:
        print("\n=== Ruff warnings ===")
        per_file_counts = Counter()

        for d in diagnostics:
            file_path = d.get("filename", "<unknown>")
            per_file_counts[file_path] += 1

            loc = d.get("location", {}) or {}
            row = loc.get("row", "?")
            col = loc.get("column", "?")

            code = d.get("code", "")
            message = d.get("message", "")
            print(f"{file_path}:{row}:{col}: {code} {message}")

        print("\n=== Ruff warnings per file ===")
        for file_path, count in per_file_counts.most_common():
            print(f"{file_path}: {count}")

        warnings.warn(f"{total_count} Ruff warnings remaining")

    assert total_count < 100, "Too many Ruff warnings found, improve code quality to pass test."
