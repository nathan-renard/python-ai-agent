import os
import sys
import tempfile
from collections.abc import Callable

from config import MAX_CHARS
from functions.get_file_content import get_file_content

# chmod can't block root, so the permission tests are skipped when running as root
IS_ROOT: bool = hasattr(os, "geteuid") and os.geteuid() == 0


def check(label: str, passed: bool) -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}: {label}")
    return passed


def truncation_message(file_path: str) -> str:
    return f'[...File "{file_path}" truncated at {MAX_CHARS} characters]'


def outside_error(file_path: str) -> str:
    return f'Error: Cannot read "{file_path}" as it is outside the permitted working directory'


def not_found_error(file_path: str) -> str:
    return f'Error: File not found or is not a regular file: "{file_path}"'


def make_fixture(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def make_sandbox(tmp_root: str) -> tuple[str, str]:
    # project/ is the working dir; outside/ and project_evil/ are siblings it must never reach
    working_dir: str = os.path.join(tmp_root, "project")
    outside_dir: str = os.path.join(tmp_root, "outside")
    make_fixture(os.path.join(working_dir, "root.txt"), "root content")
    make_fixture(os.path.join(working_dir, "sub", "nested.txt"), "nested content")
    make_fixture(os.path.join(outside_dir, "secret.txt"), "secret content")
    make_fixture(os.path.join(tmp_root, "project_evil", "x.txt"), "evil content")
    return working_dir, outside_dir


def test_course_cases() -> list[bool]:
    # Use cases: the agent reading the calculator project it has been asked to work on
    print("=== use cases: calculator/ (course cases) ===")
    results: list[bool] = []

    lorem: str = get_file_content("calculator", "lorem.txt")
    print(f"lorem.txt length: {len(lorem)}")
    print(f"lorem.txt truncated: {'truncated' in lorem}")
    results.append(check("lorem.txt is readable", not lorem.startswith("Error:")))

    # (file_path, what should happen, test on the result)
    cases: list[tuple[str, str, Callable[[str], bool]]] = [
        ("main.py", "returns the source", lambda r: "def main()" in r),
        ("pkg/calculator.py", "returns the source", lambda r: "class Calculator" in r),
        ("/bin/cat", "is refused as outside", lambda r: r == outside_error("/bin/cat")),
        ("pkg/does_not_exist.py", "is reported missing",
         lambda r: r == not_found_error("pkg/does_not_exist.py")),
    ]
    for file_path, label, passes in cases:
        print(f"=== {file_path} ===")
        result: str = get_file_content("calculator", file_path)
        print(result)
        results.append(check(f"{file_path} {label}", passes(result)))

    return results


def test_equivalence_partitions() -> list[bool]:
    print("=== equivalence partitioning ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)
        make_fixture(os.path.join(working_dir, "my notes", "café ☕.txt"), "unicode content")

        # (label, file_path, expected result)
        cases: list[tuple[str, str, str]] = [
            # Valid: regular files inside the working dir
            ("file in root", "root.txt", "root content"),
            ("file in subdir", "sub/nested.txt", "nested content"),
            ("absolute path inside working dir",
             os.path.join(working_dir, "root.txt"), "root content"),
            ("spaces and unicode in name", "my notes/café ☕.txt", "unicode content"),
            # Invalid: resolves outside the working dir
            ("absolute path outside", os.path.join(outside_dir, "secret.txt"),
             outside_error(os.path.join(outside_dir, "secret.txt"))),
            ("relative escape", "../outside/secret.txt", outside_error("../outside/secret.txt")),
            # Invalid: inside, but not a regular file
            ("missing file", "missing.txt", not_found_error("missing.txt")),
            ("directory", "sub", not_found_error("sub")),
        ]
        for label, file_path, expected in cases:
            result: str = get_file_content(working_dir, file_path)
            results.append(check(f"path: {label}", result == expected))

    return results


def test_boundaries() -> list[bool]:
    print("=== boundary values ===")
    results: list[bool] = []

    # (file name, content, should it be truncated?)
    cases: list[tuple[str, str, bool]] = [
        ("empty.txt", "", False),
        ("under_max.txt", "a" * (MAX_CHARS - 1), False),
        ("exact_max.txt", "a" * MAX_CHARS, False),
        ("over_max.txt", "a" * (MAX_CHARS + 1), True),
        # Multi-byte chars: the limit must count characters, not bytes
        ("multibyte_over_max.txt", "é" * (MAX_CHARS + 1), True),
    ]

    # Temp dir keeps calculator/ clean and makes sizes follow MAX_CHARS
    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, _ = make_sandbox(tmp_root)

        for name, content, should_truncate in cases:
            make_fixture(os.path.join(working_dir, name), content)
            result: str = get_file_content(working_dir, name)
            message: str = truncation_message(name)
            truncated: bool = result.endswith(message)
            expected_len: int = MAX_CHARS + len(message) if should_truncate else len(content)

            print(f"  {name}: {len(content)} chars in, {len(result)} chars out, truncated={truncated}")
            results.append(
                check(f"{name} truncated={should_truncate}", truncated == should_truncate)
            )
            results.append(check(f"{name} length={expected_len}", len(result) == expected_len))

        # Paths that land exactly on the working dir, just inside it, or just outside it
        edge_cases: list[tuple[str, str]] = [
            (".", not_found_error(".")),
            ("", not_found_error("")),
            ("sub/../root.txt", "root content"),
            ("../project/root.txt", "root content"),
            ("..", outside_error("..")),
            # Shares the "project" prefix, so a plain startswith() check would let it through
            ("../project_evil/x.txt", outside_error("../project_evil/x.txt")),
        ]
        for file_path, expected in edge_cases:
            result = get_file_content(working_dir, file_path)
            results.append(check(f"path edge {file_path!r}", result == expected))

    return results


def test_decision_table() -> list[bool]:
    # | Rule | inside wd | regular file | over MAX_CHARS | expected                         |
    # |------|-----------|--------------|----------------|----------------------------------|
    # | R1   | N         | Y            | -              | outside error, content not shown |
    # | R2   | N         | N            | -              | outside error, not "not found"   |
    # | R3   | Y         | N            | -              | not found error                  |
    # | R4   | Y         | Y            | N              | full content                     |
    # | R5   | Y         | Y            | Y              | first MAX_CHARS + truncation note |
    print("=== decision table ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, _ = make_sandbox(tmp_root)
        long_content: str = "b" * (MAX_CHARS + 50)
        make_fixture(os.path.join(working_dir, "long.txt"), long_content)

        result: str = get_file_content(working_dir, "../outside/secret.txt")
        results.append(check("R1 outside, file exists: refused",
                             result == outside_error("../outside/secret.txt")))

        result = get_file_content(working_dir, "../outside/missing.txt")
        results.append(check("R2 outside, file missing: outside check wins",
                             result == outside_error("../outside/missing.txt")))

        result = get_file_content(working_dir, "missing.txt")
        results.append(check("R3 inside, not a file: not found",
                             result == not_found_error("missing.txt")))

        result = get_file_content(working_dir, "root.txt")
        results.append(check("R4 inside, short file: full content", result == "root content"))

        result = get_file_content(working_dir, "long.txt")
        results.append(check("R5 inside, long file: truncated",
                             result == long_content[:MAX_CHARS] + truncation_message("long.txt")))

    return results


def test_error_guessing() -> list[bool]:
    print("=== error guessing ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)

        # Non-UTF-8 bytes can't be decoded: should come back as an error string, not raise
        with open(os.path.join(working_dir, "binary.bin"), "wb") as f:
            f.write(b"\xff\xfe\x00\x81" * 10)
        result: str = get_file_content(working_dir, "binary.bin")
        print(f"  binary.bin: {result}")
        results.append(check("binary.bin returns an error string", result.startswith("Error:")))

        # A null byte in the path
        result = get_file_content(working_dir, "bad\x00name.txt")
        results.append(check("null byte in the path: error string", result.startswith("Error:")))

        # Symlinks inside the working dir that point outside it
        os.symlink(outside_dir, os.path.join(working_dir, "link_dir"))
        os.symlink(os.path.join(outside_dir, "secret.txt"), os.path.join(working_dir, "link_file.txt"))
        result = get_file_content(working_dir, "link_dir/secret.txt")
        results.append(check("a symlinked dir can't be used to escape",
                             result == outside_error("link_dir/secret.txt")))
        result = get_file_content(working_dir, "link_file.txt")
        results.append(check("a symlinked file can't expose a file outside",
                             result == outside_error("link_file.txt")))

        if IS_ROOT:
            print("  SKIP: permission tests (running as root)")
        else:
            unreadable_path: str = os.path.join(working_dir, "root.txt")
            os.chmod(unreadable_path, 0o000)
            try:
                result = get_file_content(working_dir, "root.txt")
                results.append(check("unreadable file: error string", result.startswith("Error:")))
            finally:
                # Restore permissions so the temp dir can be cleaned up
                os.chmod(unreadable_path, 0o644)

    return results


def main() -> None:
    results: list[bool] = []
    results += test_course_cases()
    results += test_equivalence_partitions()
    results += test_boundaries()
    results += test_decision_table()
    results += test_error_guessing()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    # A non-zero exit lets CI tell a failing run from a passing one
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
