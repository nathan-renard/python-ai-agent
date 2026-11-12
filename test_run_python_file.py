import os
import signal
import tempfile
import time
from unittest.mock import patch

from config import TIMEOUT_SECONDS
from functions.run_python_file import run_python_file
from functions.write_file import write_file

# chmod can't block root, so the permission tests are skipped when running as root
IS_ROOT: bool = hasattr(os, "geteuid") and os.geteuid() == 0

# Prints the arguments the script received, so tests can see exactly what was passed
ARGV_SCRIPT: str = "import sys\nprint(sys.argv[1:])\n"


def check(label: str, passed: bool) -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}: {label}")
    return passed


def outside_error(file_path: str) -> str:
    return f'Error: Cannot execute "{file_path}" as it is outside the permitted working directory'


def missing_error(file_path: str) -> str:
    return f'Error: "{file_path}" does not exist or is not a regular file'


def not_python_error(file_path: str) -> str:
    return f'Error: "{file_path}" is not a Python file'


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


def make_marker_script(path: str, marker_path: str) -> None:
    # A script that leaves a marker file behind, so tests can prove it never ran
    make_fixture(path, f"open({marker_path!r}, 'w').close()\nprint('ran')\n")


def test_course_cases() -> list[bool]:
    print("=== course cases: calculator/ ===")
    results: list[bool] = []

    result: str = run_python_file("calculator", "main.py")
    print(result)
    results.append(check("main.py with no args prints the usage instructions",
                         result == "STDOUT:\nCalculator App\n"
                                   'Usage: python main.py "<expression>"\n'
                                   'Example: python main.py "3 + 5"\n'))

    result = run_python_file("calculator", "main.py", ["3 + 5"])
    print(result)
    results.append(check('main.py "3 + 5" renders a result of 8',
                         result == 'STDOUT:\n{\n  "expression": "3 + 5",\n  "result": 8\n}\n'))

    # unittest reports on stderr even when every test passes
    result = run_python_file("calculator", "tests.py")
    print(result)
    results.append(check("tests.py runs all 9 tests and they pass",
                         result.startswith("STDERR:\n") and "Ran 9 tests" in result
                         and result.endswith("OK\n") and "Process exited" not in result))

    # (file_path, expected error)
    error_cases: list[tuple[str, str]] = [
        ("../main.py", outside_error("../main.py")),
        ("nonexistent.py", missing_error("nonexistent.py")),
        ("lorem.txt", not_python_error("lorem.txt")),
    ]
    for file_path, expected in error_cases:
        result = run_python_file("calculator", file_path)
        print(result)
        results.append(check(f"{file_path} returns the expected error", result == expected))

    return results


def test_equivalence_partitions() -> list[bool]:
    print("=== equivalence partitioning ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)
        make_fixture(os.path.join(working_dir, "run.py"), "print('ran run.py')\n")
        make_fixture(os.path.join(working_dir, "sub", "nested.py"), "print('ran nested')\n")
        make_fixture(os.path.join(working_dir, "my scripts", "café ☕.py"), "print('ran unicode')\n")
        make_fixture(os.path.join(working_dir, "argv.py"), ARGV_SCRIPT)
        make_fixture(os.path.join(working_dir, "notes.txt"), "print('should not run')\n")
        make_fixture(os.path.join(working_dir, "README"), "print('should not run')\n")
        make_fixture(os.path.join(outside_dir, "evil.py"), "print('ran evil')\n")

        # Valid file_path classes: each one should run
        valid_paths: list[tuple[str, str, str]] = [
            ("script in the root", "run.py", "STDOUT:\nran run.py\n"),
            ("script in a subdirectory", "sub/nested.py", "STDOUT:\nran nested\n"),
            ("absolute path inside working dir", os.path.join(working_dir, "run.py"),
             "STDOUT:\nran run.py\n"),
            ("spaces and unicode in the name", "my scripts/café ☕.py", "STDOUT:\nran unicode\n"),
        ]
        for label, file_path, expected in valid_paths:
            result: str = run_python_file(working_dir, file_path)
            results.append(check(f"path: {label}", result == expected))

        # Invalid file_path classes: each one gets its own error and nothing runs
        invalid_paths: list[tuple[str, str, str]] = [
            ("relative escape", "../outside/evil.py", outside_error("../outside/evil.py")),
            ("absolute path outside", os.path.join(outside_dir, "evil.py"),
             outside_error(os.path.join(outside_dir, "evil.py"))),
            ("missing file", "missing.py", missing_error("missing.py")),
            ("directory", "sub", missing_error("sub")),
            ("text file", "notes.txt", not_python_error("notes.txt")),
            ("file with no extension", "README", not_python_error("README")),
        ]
        for label, file_path, expected in invalid_paths:
            result = run_python_file(working_dir, file_path)
            results.append(check(f"path: {label}", result == expected))

        # working_directory classes: absolute (above), relative (course cases), trailing slash
        result = run_python_file(working_dir + os.sep, "run.py")
        results.append(check("working dir: trailing slash", result == "STDOUT:\nran run.py\n"))

        # args classes; the expected text is how Python prints the list the script received
        args_cases: list[tuple[str, list[str] | None, list[str]]] = [
            ("None", None, []),
            ("empty list", [], []),
            ("one arg", ["one"], ["one"]),
            ("several args", ["a", "b", "c"], ["a", "b", "c"]),
            ("arg with spaces stays one arg", ["3 + 5"], ["3 + 5"]),
            ("unicode arg", ["héllo ☕"], ["héllo ☕"]),
        ]
        for label, args, received in args_cases:
            result = run_python_file(working_dir, "argv.py", args)
            results.append(check(f"args: {label}", result == f"STDOUT:\n{received!r}\n"))

    return results


def test_boundaries() -> list[bool]:
    print("=== boundary values ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, _ = make_sandbox(tmp_root)
        make_fixture(os.path.join(working_dir, "run.py"), "print('ran')\n")
        make_fixture(os.path.join(working_dir, "argv.py"), ARGV_SCRIPT)

        # Paths that land exactly on the working dir, just inside it, or just outside it
        path_cases: list[tuple[str, str]] = [
            (".", missing_error(".")),
            ("", missing_error("")),
            ("sub/../run.py", "STDOUT:\nran\n"),
            ("../project/run.py", "STDOUT:\nran\n"),
            ("..", outside_error("..")),
            # Shares the "project" prefix, so a plain startswith() check would let it through
            ("../project_evil/x.py", outside_error("../project_evil/x.py")),
        ]
        for file_path, expected in path_cases:
            result: str = run_python_file(working_dir, file_path)
            results.append(check(f"path edge {file_path!r}", result == expected))

        # File names right around ".py"; the extension check is case-sensitive
        name_cases: list[tuple[str, bool]] = [
            ("plain.py", True),
            (".py", True),
            ("double.py.txt", False),
            ("compiled.pyc", False),
            ("upper.PY", False),
            ("nodotpy", False),
        ]
        for name, should_run in name_cases:
            make_fixture(os.path.join(working_dir, "names", name), "print('ran')\n")
            file_path = f"names/{name}"
            expected = "STDOUT:\nran\n" if should_run else not_python_error(file_path)
            result = run_python_file(working_dir, file_path)
            results.append(check(f"file name {name!r} {'runs' if should_run else 'is refused'}",
                                 result == expected))

        # Argument edges: a single empty string is still an argument; many args all arrive
        many_args: list[str] = [str(i) for i in range(100)]
        arg_cases: list[tuple[str, list[str]]] = [
            ("one empty-string arg", [""]),
            ("100 args", many_args),
        ]
        for label, args in arg_cases:
            result = run_python_file(working_dir, "argv.py", args)
            results.append(check(f"args: {label}", result == f"STDOUT:\n{args!r}\n"))

        # Exit codes on either side of zero, up to the largest a process can return
        for code in [0, 1, 2, 255]:
            make_fixture(os.path.join(working_dir, "exit.py"), f"import sys\nsys.exit({code})\n")
            result = run_python_file(working_dir, "exit.py")
            expected = ("No output produced" if code == 0
                        else f"Process exited with code {code}\nNo output produced")
            results.append(check(f"exit code {code}", result == expected))

        # Output size around nothing: an empty script, one char, and a lone newline
        output_cases: list[tuple[str, str, str]] = [
            ("empty script", "", "No output produced"),
            ("one char, no newline", "import sys\nsys.stdout.write('x')\n", "STDOUT:\nx"),
            ("a lone newline still counts as output", "print()\n", "STDOUT:\n\n"),
        ]
        for label, code_text, expected in output_cases:
            make_fixture(os.path.join(working_dir, "out.py"), code_text)
            result = run_python_file(working_dir, "out.py")
            results.append(check(f"output: {label}", result == expected))

        # Timeout: shrink it to 1s so these checks don't wait the real 30s
        results.append(check("config timeout is 30 seconds", TIMEOUT_SECONDS == 30))
        make_fixture(os.path.join(working_dir, "quick.py"),
                     "import time\ntime.sleep(0.2)\nprint('done')\n")
        make_fixture(os.path.join(working_dir, "slow.py"),
                     "import time\ntime.sleep(10)\nprint('done')\n")
        with patch("functions.run_python_file.TIMEOUT_SECONDS", 1):
            result = run_python_file(working_dir, "quick.py")
            results.append(check("finishes inside the timeout", result == "STDOUT:\ndone\n"))

            start: float = time.monotonic()
            result = run_python_file(working_dir, "slow.py")
            elapsed: float = time.monotonic() - start
            results.append(check("runs past the timeout: stopped with an error",
                                 result.startswith("Error: executing Python file:")
                                 and "timed out" in result and elapsed < 5))

    return results


def test_decision_table() -> list[bool]:
    # Validation: which check wins when several could fail
    # | Rule | inside wd | regular file | ends .py | expected                 |
    # |------|-----------|--------------|----------|--------------------------|
    # | V1   | N         | N            | N        | outside error            |
    # | V2   | Y         | N (missing)  | Y        | missing error            |
    # | V3   | Y         | N (missing)  | N        | missing error, not .py   |
    # | V4   | Y         | N (dir)      | Y        | missing error            |
    # | V5   | Y         | Y            | N        | not a Python file error  |
    # | V6   | Y         | Y            | Y        | runs                     |
    #
    # Output: how the result string is built from the finished process
    # | Rule | exit code | stdout | stderr | expected                                     |
    # |------|-----------|--------|--------|----------------------------------------------|
    # | O1   | 0         | N      | N      | No output produced                           |
    # | O2   | 0         | Y      | N      | STDOUT only                                  |
    # | O3   | 0         | N      | Y      | STDERR only                                  |
    # | O4   | 0         | Y      | Y      | STDOUT then STDERR                           |
    # | O5   | 3         | N      | N      | exit code line, No output produced           |
    # | O6   | 3         | Y      | N      | exit code line, STDOUT                       |
    # | O7   | 3         | N      | Y      | exit code line, STDERR                       |
    # | O8   | 3         | Y      | Y      | exit code line, STDOUT then STDERR           |
    print("=== decision table ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, _ = make_sandbox(tmp_root)
        make_fixture(os.path.join(working_dir, "run.py"), "print('ran')\n")
        make_fixture(os.path.join(working_dir, "notes.txt"), "print('ran')\n")
        os.makedirs(os.path.join(working_dir, "pkg.py"))

        validation_rules: list[tuple[str, str, str]] = [
            ("V1 outside, missing, not .py: outside check wins", "../outside/missing.txt",
             outside_error("../outside/missing.txt")),
            ("V2 inside, missing .py", "missing.py", missing_error("missing.py")),
            ("V3 inside, missing, not .py: existence check wins", "missing.txt",
             missing_error("missing.txt")),
            ("V4 a directory named like a script", "pkg.py", missing_error("pkg.py")),
            ("V5 inside, real file, not .py", "notes.txt", not_python_error("notes.txt")),
            ("V6 inside, real .py file: runs", "run.py", "STDOUT:\nran\n"),
        ]
        for label, file_path, expected in validation_rules:
            result: str = run_python_file(working_dir, file_path)
            results.append(check(label, result == expected))

        # (rule, exit code, stdout text, stderr text, expected)
        output_rules: list[tuple[str, int, str, str, str]] = [
            ("O1", 0, "", "", "No output produced"),
            ("O2", 0, "out", "", "STDOUT:\nout"),
            ("O3", 0, "", "err", "STDERR:\nerr"),
            ("O4", 0, "out", "err", "STDOUT:\nout\nSTDERR:\nerr"),
            ("O5", 3, "", "", "Process exited with code 3\nNo output produced"),
            ("O6", 3, "out", "", "Process exited with code 3\nSTDOUT:\nout"),
            ("O7", 3, "", "err", "Process exited with code 3\nSTDERR:\nerr"),
            ("O8", 3, "out", "err", "Process exited with code 3\nSTDOUT:\nout\nSTDERR:\nerr"),
        ]
        for rule, code, stdout, stderr, expected in output_rules:
            make_fixture(os.path.join(working_dir, "rule.py"),
                         f"import sys\nsys.stdout.write({stdout!r})\n"
                         f"sys.stderr.write({stderr!r})\nsys.exit({code})\n")
            result = run_python_file(working_dir, "rule.py")
            results.append(check(f"{rule} exit={code} stdout={bool(stdout)} stderr={bool(stderr)}",
                                 result == expected))

    return results


def test_use_cases() -> list[bool]:
    print("=== use cases: agent workflows ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)

        # UC1: the agent runs the tests, sees a failure, fixes the code, and reruns them
        tests: str = (
            "import unittest\nfrom mathlib import add\n\n"
            "class TestAdd(unittest.TestCase):\n"
            "    def test_add(self) -> None:\n"
            "        self.assertEqual(add(2, 3), 5)\n\n"
            "unittest.main()\n"
        )
        write_file(working_dir, "mathlib.py", "def add(a: int, b: int) -> int:\n    return a - b\n")
        write_file(working_dir, "test_mathlib.py", tests)
        result: str = run_python_file(working_dir, "test_mathlib.py")
        results.append(check("UC1 a failing test run reports the failure and exit code 1",
                             result.startswith("Process exited with code 1\n")
                             and "FAILED (failures=1)" in result))

        # The fix changes the file size: a same-size rewrite in the same second can reuse
        # Python's cached bytecode for mathlib and still run the old code
        write_file(working_dir, "mathlib.py",
                   "def add(a: int, b: int) -> int:\n    return a + b  # fixed\n")
        result = run_python_file(working_dir, "test_mathlib.py")
        results.append(check("UC1 after the fix the same tests pass",
                             "Process exited" not in result and result.endswith("OK\n")))

        # UC2: the agent uses the calculator to evaluate an expression
        result = run_python_file("calculator", "main.py", ["3 * 4 + 5"])
        results.append(check("UC2 the calculator evaluates 3 * 4 + 5 to 17",
                             '"result": 17' in result))

        # UC3: the agent runs a script that crashes and gets the traceback to debug with
        write_file(working_dir, "crash.py", "print('starting')\nprint(1 / 0)\n")
        result = run_python_file(working_dir, "crash.py")
        results.append(check("UC3 a crash returns stdout so far plus the traceback",
                             result.startswith("Process exited with code 1\nSTDOUT:\nstarting\n")
                             and "ZeroDivisionError: division by zero" in result))

        # UC4: a nested script runs from the working dir, so relative paths start there
        write_file(working_dir, "data.txt", "hello data")
        write_file(working_dir, "sub/report.py",
                   "import os\nprint(os.getcwd())\nprint(open('data.txt').read())\n")
        result = run_python_file(working_dir, "sub/report.py")
        results.append(check("UC4 the script's cwd is the working dir",
                             result == f"STDOUT:\n{os.path.realpath(working_dir)}\nhello data\n"))

        # UC5: the agent is talked into running a script outside its sandbox
        marker: str = os.path.join(outside_dir, "ran.marker")
        make_marker_script(os.path.join(outside_dir, "evil.py"), marker)
        result = run_python_file(working_dir, "../outside/evil.py")
        results.append(check("UC5 a script outside the sandbox never runs",
                             result == outside_error("../outside/evil.py")
                             and not os.path.exists(marker)))

    return results


def test_error_guessing() -> list[bool]:
    print("=== error guessing ===")
    results: list[bool] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        working_dir, outside_dir = make_sandbox(tmp_root)
        make_fixture(os.path.join(working_dir, "argv.py"), ARGV_SCRIPT)
        make_fixture(os.path.join(working_dir, "notes.txt"), "print('ran')\n")
        marker: str = os.path.join(outside_dir, "ran.marker")
        make_marker_script(os.path.join(outside_dir, "evil.py"), marker)

        # Shell syntax in args: no shell is involved, so it arrives as plain text
        shell_args: list[str] = ["; touch pwned.txt", "$(touch pwned.txt)", "&& touch pwned.txt"]
        result: str = run_python_file(working_dir, "argv.py", shell_args)
        results.append(check("shell syntax in args is passed literally, never run",
                             result == f"STDOUT:\n{shell_args!r}\n"
                             and not os.path.exists(os.path.join(working_dir, "pwned.txt"))))

        # Interpreter flags in args go to the script, not to python
        flag_args: list[str] = ["-c", "print('hijacked')"]
        result = run_python_file(working_dir, "argv.py", flag_args)
        results.append(check("python flags in args reach the script, not the interpreter",
                             result == f"STDOUT:\n{flag_args!r}\n"))

        # Symlinks inside the working dir that point outside it
        os.symlink(os.path.join(outside_dir, "evil.py"), os.path.join(working_dir, "escape.py"))
        os.symlink(outside_dir, os.path.join(working_dir, "link_dir"))
        for file_path in ["escape.py", "link_dir/evil.py"]:
            result = run_python_file(working_dir, file_path)
            results.append(check(f"symlink {file_path} can't run a script outside",
                                 result == outside_error(file_path) and not os.path.exists(marker)))

        # A .py symlink that points at a text file
        os.symlink(os.path.join(working_dir, "notes.txt"), os.path.join(working_dir, "fake.py"))
        result = run_python_file(working_dir, "fake.py")
        results.append(check("a .py symlink to a .txt file is refused",
                             result == not_python_error("fake.py")))

        # A null byte in the path makes the os calls raise ValueError
        result = run_python_file(working_dir, "bad\x00name.py")
        results.append(check("null byte in the path: error string", result.startswith("Error:")))

        # A working dir that doesn't exist has no files in it
        result = run_python_file(os.path.join(tmp_root, "no_such_dir"), "main.py")
        results.append(check("missing working dir: missing file error",
                             result == missing_error("main.py")))

        # A script that isn't valid Python
        make_fixture(os.path.join(working_dir, "broken.py"), "def oops(:\n")
        result = run_python_file(working_dir, "broken.py")
        results.append(check("syntax error: exit code 1 and the SyntaxError",
                             result.startswith("Process exited with code 1\n")
                             and "SyntaxError" in result))

        # A process killed by a signal gets a negative return code
        make_fixture(os.path.join(working_dir, "killed.py"),
                     f"import os\nos.kill(os.getpid(), {int(signal.SIGTERM)})\n")
        result = run_python_file(working_dir, "killed.py")
        results.append(check("killed by SIGTERM: negative exit code",
                             result == f"Process exited with code {-int(signal.SIGTERM)}\n"
                                       "No output produced"))

        # Bytes that aren't valid UTF-8 can't be decoded to text
        make_fixture(os.path.join(working_dir, "binary.py"),
                     "import sys\nsys.stdout.buffer.write(b'\\xff\\xfe')\n")
        result = run_python_file(working_dir, "binary.py")
        results.append(check("non-UTF-8 output: error string, no crash",
                             result.startswith("Error: executing Python file:")))

        if IS_ROOT:
            print("  SKIP: permission test (running as root)")
        else:
            locked_path: str = os.path.join(working_dir, "locked.py")
            make_fixture(locked_path, "print('ran')\n")
            os.chmod(locked_path, 0o000)
            try:
                result = run_python_file(working_dir, "locked.py")
                results.append(check("unreadable script: python's own error comes back",
                                     result.startswith("Process exited with code 2\n")
                                     and "Permission denied" in result))
            finally:
                os.chmod(locked_path, 0o644)

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
