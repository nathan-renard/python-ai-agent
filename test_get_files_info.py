import os
import tempfile
from collections.abc import Callable

from functions.get_files_info import get_files_info

# chmod can't block root, so the permission tests are skipped when running as root
IS_ROOT: bool = hasattr(os, "geteuid") and os.geteuid() == 0


def check(label: str, passed: bool) -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}: {label}")
    return passed


def outside_error(directory: str) -> str:
    return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'


def not_dir_error(directory: str) -> str:
    return f'Error: "{directory}" is not a directory'


def entry(name: str, size: int, is_dir: bool) -> str:
    return f"- {name}: file_size={size} bytes, is_dir={is_dir}"


def make_fixture(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def make_sandbox(tmp_root: str) -> tuple[str, str]:
    # project/ is the working dir; outside/ and project_evil/ are siblings it must never reach
    working_dir: str = os.path.join(tmp_root, "project")
    outside_dir: str = os.path.join(tmp_root, "outside")
    make_fixture(os.path.join(working_dir, "root.txt"), "hello")
    make_fixture(os.path.join(working_dir, "sub", "nested", "deep.txt"), "deep")
    make_fixture(os.path.join(outside_dir, "secret.txt"), "secret")
    make_fixture(os.path.join(tmp_root, "project_evil", "x.txt"), "evil")
    return working_dir, outside_dir


def root_listing(working_dir: str) -> str:
    # What listing the working dir itself should produce
    sub_size: int = os.path.getsize(os.path.join(working_dir, "sub"))
    return "\n".join([entry("root.txt", 5, False), entry("sub", sub_size, True)])


def test_course_cases() -> list[bool]:
    # Use cases: the agent exploring the calculator project it has been asked to work on
    print("=== use cases: calculator/ (course cases) ===")
    results: list[bool] = []

    def has_entry(listing: str, rel_path: str, is_dir: bool) -> bool:
        size: int = os.path.getsize(os.path.join("calculator", rel_path))
        return entry(os.path.basename(rel_path), size, is_dir) in listing.splitlines()

    # (directory, what should happen, test on the result)
    cases: list[tuple[str, str, Callable[[str], bool]]] = [
        (".", "lists main.py and pkg/",
         lambda r: has_entry(r, "main.py", False) and has_entry(r, "pkg", True)),
        ("pkg", "lists calculator.py and render.py",
         lambda r: has_entry(r, "pkg/calculator.py", False) and has_entry(r, "pkg/render.py", False)),
        ("/bin", "is refused as outside", lambda r: r == outside_error("/bin")),
        ("../", "is refused as outside", lambda r: r == outside_error("../")),
        ("main.py", "is reported as not a directory", lambda r: r == not_dir_error("main.py")),
    ]
    for directory, label, passes in cases:
        print(f"=== {directory} ===")
        result: str = get_files_info("calculator", directory)
        print(result)
        results.append(check(f"{directory} {label}", passes(result)))

    return results


def test_equivalence_partitions() -> list[bool]:
    print("=== equivalence partitioning ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)
        expected_root: str = root_listing(working_dir)
        nested_size: int = os.path.getsize(os.path.join(working_dir, "sub", "nested"))

        # (label, directory, expected result)
        cases: list[tuple[str, str, str]] = [
            # Valid: directories inside the working dir
            ("working dir itself", ".", expected_root),
            ("subdir", "sub", entry("nested", nested_size, True)),
            ("nested subdir", "sub/nested", entry("deep.txt", 4, False)),
            ("absolute path inside working dir", working_dir, expected_root),
            # Invalid: resolves outside the working dir
            ("absolute path outside", outside_dir, outside_error(outside_dir)),
            ("relative escape", "../outside", outside_error("../outside")),
            # Invalid: inside, but not a directory
            ("a file", "root.txt", not_dir_error("root.txt")),
            ("missing path", "missing", not_dir_error("missing")),
        ]
        for label, directory, expected in cases:
            result: str = get_files_info(working_dir, directory)
            results.append(check(f"directory: {label}", result == expected))

        result = get_files_info(working_dir)
        results.append(check("directory omitted: defaults to the working dir",
                             result == expected_root))

    return results


def test_boundaries() -> list[bool]:
    print("=== boundary values ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, _ = make_sandbox(tmp_root)
        expected_root: str = root_listing(working_dir)

        # Paths that land exactly on the working dir, just inside it, or just outside it.
        # Runs before the fixtures below are added, since they'd change the root listing.
        edge_cases: list[tuple[str, str]] = [
            ("", expected_root),
            ("sub/..", expected_root),
            ("../project", expected_root),
            ("..", outside_error("..")),
            # Shares the "project" prefix, so a plain startswith() check would let it through
            ("../project_evil", outside_error("../project_evil")),
        ]
        for directory, expected in edge_cases:
            result: str = get_files_info(working_dir, directory)
            results.append(check(f"path edge {directory!r}", result == expected))

        os.makedirs(os.path.join(working_dir, "empty"))
        make_fixture(os.path.join(working_dir, "one", "only.txt"), "x")
        make_fixture(os.path.join(working_dir, "sizes", "zero.txt"), "")
        make_fixture(os.path.join(working_dir, "sizes", "one_byte.txt"), "x")

        # Entry count: 0 and 1
        result = get_files_info(working_dir, "empty")
        results.append(check("0 entries: empty string", result == ""))
        result = get_files_info(working_dir, "one")
        results.append(check("1 entry: exactly one line", result == entry("only.txt", 1, False)))

        # File size: 0 and 1 byte
        result = get_files_info(working_dir, "sizes")
        results.append(check("file sizes of 0 and 1 byte",
                             result == "\n".join([entry("one_byte.txt", 1, False),
                                                  entry("zero.txt", 0, False)])))

    return results


def test_decision_table() -> list[bool]:
    # | Rule | inside wd | exists | is a dir | expected                       |
    # |------|-----------|--------|----------|--------------------------------|
    # | R1   | N         | Y      | Y        | outside error, nothing listed  |
    # | R2   | N         | N      | -        | outside error, not "not a dir" |
    # | R3   | Y         | N      | -        | not a directory error          |
    # | R4   | Y         | Y      | N        | not a directory error          |
    # | R5   | Y         | Y      | Y        | sorted listing, is_dir correct |
    print("=== decision table ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, _ = make_sandbox(tmp_root)
        order_dir: str = os.path.join(working_dir, "order")
        make_fixture(os.path.join(order_dir, "zeta.txt"), "zz")
        make_fixture(os.path.join(order_dir, "alpha.txt"), "a")
        os.makedirs(os.path.join(order_dir, "mid"))
        mid_size: int = os.path.getsize(os.path.join(order_dir, "mid"))

        result: str = get_files_info(working_dir, "../outside")
        results.append(check("R1 outside, existing dir: refused",
                             result == outside_error("../outside")))

        result = get_files_info(working_dir, "../outside/missing")
        results.append(check("R2 outside, missing: outside check wins",
                             result == outside_error("../outside/missing")))

        result = get_files_info(working_dir, "missing")
        results.append(check("R3 inside, missing: not a directory",
                             result == not_dir_error("missing")))

        result = get_files_info(working_dir, "root.txt")
        results.append(check("R4 inside, a file: not a directory",
                             result == not_dir_error("root.txt")))

        result = get_files_info(working_dir, "order")
        expected: str = "\n".join([entry("alpha.txt", 1, False),
                                   entry("mid", mid_size, True),
                                   entry("zeta.txt", 2, False)])
        results.append(check("R5 inside, a dir: sorted listing with is_dir", result == expected))

    return results


def test_error_guessing() -> list[bool]:
    print("=== error guessing ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)

        # A null byte in the path
        result: str = get_files_info(working_dir, "bad\x00name")
        results.append(check("null byte in the path: error string", result.startswith("Error:")))

        # A symlink inside the working dir that points outside it
        os.symlink(outside_dir, os.path.join(working_dir, "link_dir"))
        result = get_files_info(working_dir, "link_dir")
        results.append(check("a symlinked dir can't be used to escape",
                             result == outside_error("link_dir")))

        # A broken symlink in the folder being listed: getsize() on it raises
        broken_dir: str = os.path.join(working_dir, "broken")
        os.makedirs(broken_dir)
        os.symlink(os.path.join(tmp_root, "gone.txt"), os.path.join(broken_dir, "dangling.txt"))
        result = get_files_info(working_dir, "broken")
        results.append(check("broken symlink: error string, no exception", result.startswith("Error:")))

        if IS_ROOT:
            print("  SKIP: permission tests (running as root)")
        else:
            locked_dir: str = os.path.join(working_dir, "sub")
            os.chmod(locked_dir, 0o000)
            try:
                result = get_files_info(working_dir, "sub")
                results.append(check("unreadable dir: error string", result.startswith("Error:")))
            finally:
                # Restore permissions so the temp dir can be cleaned up
                os.chmod(locked_dir, 0o755)

    return results


def main() -> None:
    results: list[bool] = []
    results += test_course_cases()
    results += test_equivalence_partitions()
    results += test_boundaries()
    results += test_decision_table()
    results += test_error_guessing()
    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
