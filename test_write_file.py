import os
import tempfile

from config import MAX_CHARS
from functions.get_file_content import get_file_content
from functions.get_files_info import get_files_info
from functions.write_file import write_file

# chmod can't block root, so the permission tests are skipped when running as root
IS_ROOT: bool = hasattr(os, "geteuid") and os.geteuid() == 0


def check(label: str, passed: bool) -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}: {label}")
    return passed


def success(file_path: str, content: str) -> str:
    return f'Successfully wrote to "{file_path}" ({len(content)} characters written)'


def outside_error(file_path: str) -> str:
    return f'Error: Cannot write to "{file_path}" as it is outside the permitted working directory'


def directory_error(file_path: str) -> str:
    return f'Error: Cannot write to "{file_path}" as it is a directory'


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def make_fixture(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def make_sandbox(tmp_root: str) -> tuple[str, str]:
    # project/ is the working dir; outside/ and project_evil/ are siblings it must never reach
    working_dir: str = os.path.join(tmp_root, "project")
    outside_dir: str = os.path.join(tmp_root, "outside")
    os.makedirs(os.path.join(working_dir, "sub"))
    os.makedirs(outside_dir)
    os.makedirs(os.path.join(tmp_root, "project_evil"))
    return working_dir, outside_dir


def test_course_cases() -> list[bool]:
    print("=== course cases: calculator/ ===")
    results: list[bool] = []
    lorem_path: str = os.path.join("calculator", "lorem.txt")
    morelorem_path: str = os.path.join("calculator", "pkg", "morelorem.txt")
    outside_path: str = "/tmp/temp.txt"

    # These cases write into the real calculator/ dir: snapshot it now, put it back afterwards
    with open(lorem_path, "rb") as f:
        original_lorem: bytes = f.read()
    morelorem_existed: bool = os.path.exists(morelorem_path)
    outside_existed: bool = os.path.exists(outside_path)

    # (file_path, content, expected result)
    cases: list[tuple[str, str, str]] = [
        ("lorem.txt", "wait, this isn't lorem ipsum",
         success("lorem.txt", "wait, this isn't lorem ipsum")),
        ("pkg/morelorem.txt", "lorem ipsum dolor sit amet",
         success("pkg/morelorem.txt", "lorem ipsum dolor sit amet")),
        (outside_path, "this should not be allowed", outside_error(outside_path)),
    ]
    try:
        for file_path, content, expected in cases:
            result: str = write_file("calculator", file_path, content)
            print(result)
            results.append(check(f"{file_path} returns the expected message", result == expected))

        results.append(check("lorem.txt now holds the new text",
                             read_text(lorem_path) == "wait, this isn't lorem ipsum"))
        results.append(check("pkg/morelorem.txt was created",
                             read_text(morelorem_path) == "lorem ipsum dolor sit amet"))
        results.append(check("/tmp/temp.txt was not created",
                             outside_existed or not os.path.exists(outside_path)))
    finally:
        with open(lorem_path, "wb") as f:
            f.write(original_lorem)
        if not morelorem_existed and os.path.exists(morelorem_path):
            os.remove(morelorem_path)

    return results


def test_equivalence_partitions() -> list[bool]:
    print("=== equivalence partitioning ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)
        make_fixture(os.path.join(working_dir, "existing.txt"), "old content")

        # Valid file_path classes: each one should be written
        valid_paths: list[tuple[str, str]] = [
            ("new file in root", "new.txt"),
            ("existing file in root", "existing.txt"),
            ("new file in existing subdir", "sub/new.txt"),
            ("new file under missing dirs", "a/b/new.txt"),
            ("absolute path inside working dir", os.path.join(working_dir, "absolute.txt")),
            ("spaces and unicode in name", "my notes/café ☕.txt"),
        ]
        for label, file_path in valid_paths:
            content: str = f"written via {label}"
            result: str = write_file(working_dir, file_path, content)
            written: str = read_text(os.path.join(working_dir, file_path))
            results.append(check(f"path: {label}",
                                 result == success(file_path, content) and written == content))

        # Invalid class: anything that resolves outside the working dir
        outside_paths: list[tuple[str, str]] = [
            ("absolute path outside", os.path.join(outside_dir, "x.txt")),
            ("relative escape", "../outside/x.txt"),
        ]
        for label, file_path in outside_paths:
            result = write_file(working_dir, file_path, "nope")
            created: bool = os.path.exists(os.path.join(outside_dir, "x.txt"))
            results.append(check(f"path: {label}",
                                 result == outside_error(file_path) and not created))

        # Invalid class: an existing directory
        result = write_file(working_dir, "sub", "nope")
        results.append(check("path: existing directory",
                             result == directory_error("sub")
                             and os.path.isdir(os.path.join(working_dir, "sub"))))

        # Content classes; multi-byte proves the count is characters, not bytes
        content_cases: list[tuple[str, str]] = [
            ("empty", ""),
            ("single-line ASCII", "hello"),
            ("multi-line", "line 1\nline 2\n"),
            ("multi-byte unicode", "héllo wörld ☕"),
        ]
        for label, content in content_cases:
            result = write_file(working_dir, "content.txt", content)
            written = read_text(os.path.join(working_dir, "content.txt"))
            results.append(check(f"content: {label} ({len(content)} chars)",
                                 result == success("content.txt", content) and written == content))

    return results


def test_boundaries() -> list[bool]:
    print("=== boundary values ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, _ = make_sandbox(tmp_root)

        # Content length around 0 and MAX_CHARS: reading truncates there, writing must not
        lengths: list[int] = [0, 1, MAX_CHARS - 1, MAX_CHARS, MAX_CHARS + 1]
        for length in lengths:
            content: str = "a" * length
            result: str = write_file(working_dir, "sized.txt", content)
            size: int = os.path.getsize(os.path.join(working_dir, "sized.txt"))
            results.append(check(f"content length {length}",
                                 result == success("sized.txt", content) and size == length))

        # Paths that land exactly on the working dir, just inside it, or just outside it
        edge_cases: list[tuple[str, str]] = [
            (".", directory_error(".")),
            ("", directory_error("")),
            ("sub/..", directory_error("sub/..")),
            ("sub/../edge.txt", success("sub/../edge.txt", "edge")),
            ("../project/back_inside.txt", success("../project/back_inside.txt", "edge")),
            ("..", outside_error("..")),
            # Shares the "project" prefix, so a plain startswith() check would let it through
            ("../project_evil/x.txt", outside_error("../project_evil/x.txt")),
        ]
        for file_path, expected in edge_cases:
            result = write_file(working_dir, file_path, "edge")
            results.append(check(f"path edge {file_path!r}", result == expected))

        # File name length at and just over the file system's limit
        name_max: int = os.pathconf(working_dir, "PC_NAME_MAX")
        longest_name: str = "a" * name_max
        result = write_file(working_dir, longest_name, "x")
        results.append(check(f"file name of {name_max} chars (the limit)",
                             result == success(longest_name, "x")))
        result = write_file(working_dir, longest_name + "a", "x")
        results.append(check(f"file name of {name_max + 1} chars returns an error",
                             result.startswith("Error:")))

        # Nesting depth: many missing directory levels at once
        deep_path: str = "/".join(f"d{i}" for i in range(10)) + "/deep.txt"
        result = write_file(working_dir, deep_path, "deep")
        results.append(check("10 missing directory levels are created",
                             result == success(deep_path, "deep")
                             and read_text(os.path.join(working_dir, deep_path)) == "deep"))

    return results


def test_decision_table() -> list[bool]:
    # | Rule | inside wd | is a dir | parent exists | file exists | expected                       |
    # |------|-----------|----------|---------------|-------------|--------------------------------|
    # | R1   | N         | N        | N             | N           | outside error, nothing created |
    # | R2   | N         | Y        | -             | -           | outside error, not dir error   |
    # | R3   | Y         | Y        | -             | -           | directory error, dir untouched |
    # | R4   | Y         | N        | N             | N           | parents created, file written  |
    # | R5   | Y         | N        | Y             | N           | file created                   |
    # | R6   | Y         | N        | Y             | Y           | file overwritten, not appended |
    print("=== decision table ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)
        make_fixture(os.path.join(working_dir, "sub", "keep.txt"), "keep")
        make_fixture(os.path.join(working_dir, "existing.txt"), "a much longer original line")

        result: str = write_file(working_dir, "../outside/new_dir/x.txt", "x")
        results.append(check("R1 outside, missing parent: refused, no dirs created",
                             result == outside_error("../outside/new_dir/x.txt")
                             and not os.path.exists(os.path.join(outside_dir, "new_dir"))))

        result = write_file(working_dir, "../outside", "x")
        results.append(check("R2 outside and a dir: outside check wins",
                             result == outside_error("../outside")))

        result = write_file(working_dir, "sub", "x")
        results.append(check("R3 inside and a dir: refused, contents kept",
                             result == directory_error("sub")
                             and read_text(os.path.join(working_dir, "sub", "keep.txt")) == "keep"))

        result = write_file(working_dir, "new_dir/nested/x.txt", "x")
        results.append(check("R4 missing parents: created and written",
                             result == success("new_dir/nested/x.txt", "x")
                             and read_text(os.path.join(working_dir, "new_dir/nested/x.txt")) == "x"))

        result = write_file(working_dir, "sub/fresh.txt", "fresh")
        results.append(check("R5 parent exists, new file: created",
                             result == success("sub/fresh.txt", "fresh")
                             and read_text(os.path.join(working_dir, "sub/fresh.txt")) == "fresh"))

        result = write_file(working_dir, "existing.txt", "short")
        results.append(check("R6 existing file: overwritten, not appended",
                             result == success("existing.txt", "short")
                             and read_text(os.path.join(working_dir, "existing.txt")) == "short"))

    return results


def test_use_cases() -> list[bool]:
    print("=== use cases: agent workflows ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)

        # UC1: the agent creates a new module, then reads it back to confirm
        buggy: str = "def add(a: int, b: int) -> int:\n    return a - b\n"
        write_file(working_dir, "pkg/helpers.py", buggy)
        results.append(check("UC1 create a module and read it back",
                             get_file_content(working_dir, "pkg/helpers.py") == buggy))

        # UC2: the agent fixes a bug by rewriting the file it just wrote
        fixed: str = buggy.replace("a - b", "a + b")
        write_file(working_dir, "pkg/helpers.py", fixed)
        results.append(check("UC2 rewrite a file to fix a bug",
                             get_file_content(working_dir, "pkg/helpers.py") == fixed))

        # UC3: the agent lists the folder and sees the new file with its size in bytes
        listing: str = get_files_info(working_dir, "pkg")
        expected_line: str = f"- helpers.py: file_size={len(fixed.encode())} bytes, is_dir=False"
        results.append(check("UC3 the new file shows up in the listing",
                             expected_line in listing.splitlines()))

        # UC4: the agent is talked into overwriting a file outside its sandbox
        secret_path: str = os.path.join(outside_dir, "secret.txt")
        make_fixture(secret_path, "original")
        result: str = write_file(working_dir, "../outside/secret.txt", "pwned")
        results.append(check("UC4 a file outside the sandbox can't be overwritten",
                             result == outside_error("../outside/secret.txt")
                             and read_text(secret_path) == "original"))

    return results


def test_error_guessing() -> list[bool]:
    print("=== error guessing ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)
        existing_path: str = os.path.join(working_dir, "existing.txt")
        secret_path: str = os.path.join(outside_dir, "secret.txt")
        make_fixture(existing_path, "original")
        make_fixture(secret_path, "original")

        # A "parent directory" that is really a file
        result: str = write_file(working_dir, "existing.txt/child.txt", "x")
        results.append(check("parent in the path is a file: error string",
                             result.startswith("Error:")))

        # A null byte in the path makes open() raise ValueError
        result = write_file(working_dir, "bad\x00name.txt", "x")
        results.append(check("null byte in the path: error string", result.startswith("Error:")))

        # Symlinks inside the working dir that point outside it
        os.symlink(outside_dir, os.path.join(working_dir, "link_dir"))
        os.symlink(secret_path, os.path.join(working_dir, "link_file.txt"))
        result = write_file(working_dir, "link_dir/pwned.txt", "pwned")
        results.append(check("a symlinked dir can't be used to escape",
                             result == outside_error("link_dir/pwned.txt")
                             and not os.path.exists(os.path.join(outside_dir, "pwned.txt"))))
        result = write_file(working_dir, "link_file.txt", "pwned")
        results.append(check("a symlinked file can't overwrite a file outside",
                             result == outside_error("link_file.txt")
                             and read_text(secret_path) == "original"))

        if IS_ROOT:
            print("  SKIP: permission tests (running as root)")
        else:
            readonly_dir: str = os.path.join(working_dir, "sub")
            os.chmod(existing_path, 0o444)
            os.chmod(readonly_dir, 0o555)
            try:
                result = write_file(working_dir, "existing.txt", "changed")
                results.append(check("read-only file: error string, content kept",
                                     result.startswith("Error:")
                                     and read_text(existing_path) == "original"))
                result = write_file(working_dir, "sub/new.txt", "x")
                results.append(check("read-only dir: error string", result.startswith("Error:")))
            finally:
                # Restore permissions so the temp dir can be cleaned up
                os.chmod(existing_path, 0o644)
                os.chmod(readonly_dir, 0o755)

    return results


def main() -> None:
    results: list[bool] = []
    results += test_course_cases()
    results += test_equivalence_partitions()
    results += test_boundaries()
    results += test_decision_table()
    results += test_use_cases()
    results += test_error_guessing()
    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
