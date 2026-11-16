import inspect
import io
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import chdir, contextmanager, redirect_stdout
from dataclasses import dataclass
from typing import Any

from openai.types.chat import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

from call_function import available_functions, call_function, function_map
from config import MAX_CHARS
from functions.get_file_content import get_file_content
from functions.get_files_info import get_files_info
from functions.run_python_file import run_python_file

# call_function.py injects this, so the tests use it as the oracle for what each function returns
WORKING_DIR: str = "./calculator"

# Prints the arguments the script received, so tests can see exactly what was passed
ARGV_SCRIPT: str = "import sys\nprint(sys.argv[1:])\n"

EXPECTED_FUNCTIONS: list[str] = ["get_files_info", "get_file_content", "run_python_file", "write_file"]


@dataclass
class CallResult:
    message: ChatCompletionToolMessageParam | None  # what call_function returned, None if it raised
    output: str                                       # what it printed
    error: Exception | None                           # the exception it let escape, if any

    @property
    def content(self) -> Any:
        return self.message["content"] if self.message is not None else None


def check(label: str, passed: bool) -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}: {label}")
    return passed


def fn_call(name: str, arguments: str | None, call_id: str = "call_1") -> ChatCompletionMessageFunctionToolCall:
    # model_construct skips validation, so a test can send arguments=None like a sloppy provider might
    return ChatCompletionMessageFunctionToolCall(
        id=call_id,
        type="function",
        function=Function.model_construct(name=name, arguments=arguments),
    )


def call(name: str, arguments: str | None, verbose: bool = False, call_id: str = "call_1") -> CallResult:
    out: io.StringIO = io.StringIO()
    message: ChatCompletionToolMessageParam | None = None
    error: Exception | None = None
    with redirect_stdout(out):
        try:
            message = call_function(fn_call(name, arguments, call_id), verbose)
        except Exception as e:
            error = e
    return CallResult(message, out.getvalue(), error)


def announced(name: str) -> str:
    return f" - Calling function: {name}\n"


def make_fixture(path: str, content: str) -> None:
    # A bare file name has no folder part, and makedirs("") fails
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@contextmanager
def sandbox() -> Iterator[str]:
    # A throwaway folder with its own calculator/, made the current directory, so
    # "./calculator" points here and nothing in the real project gets touched
    with tempfile.TemporaryDirectory() as tmp_root, chdir(tmp_root):
        make_fixture("calculator/main.py", "print('hello from main')\n")
        make_fixture("calculator/argv.py", ARGV_SCRIPT)
        make_fixture("calculator/crash.py", "raise SystemExit('boom')\n")
        make_fixture("calculator/notes.txt", "some notes\n")
        make_fixture("calculator/empty.txt", "")
        make_fixture("calculator/pkg/helper.py", "x = 1\n")
        make_fixture("calculator/dossier été/fichier.txt", "bonjour\n")
        os.makedirs("calculator/empty_dir")
        make_fixture("secret.txt", "top secret\n")  # outside the working directory
        yield tmp_root


def schema_properties(tool: dict[str, Any]) -> dict[str, Any]:
    return tool["function"].get("parameters", {}).get("properties", {})


def test_course_cases() -> list[bool]:
    # The real calculator, read-only calls
    print("=== use cases: calculator/ ===")
    results: list[bool] = []

    # (name, arguments, oracle for the result)
    cases: list[tuple[str, str, str]] = [
        ("get_files_info", '{"directory": "."}', get_files_info(WORKING_DIR, ".")),
        ("get_files_info", '{"directory": "pkg"}', get_files_info(WORKING_DIR, "pkg")),
        ("get_file_content", '{"file_path": "main.py"}', get_file_content(WORKING_DIR, "main.py")),
        ("run_python_file", '{"file_path": "main.py"}', run_python_file(WORKING_DIR, "main.py")),
        ("run_python_file", '{"file_path": "main.py", "args": ["3 + 5"]}',
         run_python_file(WORKING_DIR, "main.py", ["3 + 5"])),
    ]
    for name, arguments, expected in cases:
        result: CallResult = call(name, arguments)
        print(result.output, end="")
        results.append(check(f"{name} {arguments}: announced, real result returned",
                             result.output == announced(name) and result.content == expected))

    result = call("run_python_file", '{"file_path": "main.py", "args": ["3 + 5"]}')
    results.append(check("run main.py '3 + 5': the model gets the rendered result",
                         '"result": 8' in str(result.content)))

    # Writing goes to a sandbox so the real calculator/ stays as it is
    with sandbox():
        result = call("write_file", '{"file_path": "README.md", "content": "hello"}')
        print(result.output, end="")
        with open("calculator/README.md", encoding="utf-8") as f:
            written: str = f.read()
        results.append(check("write_file README.md: file written inside calculator/",
                             written == "hello"
                             and result.content == 'Successfully wrote to "README.md" (5 characters written)'))

    return results


def test_equivalence_partitions() -> list[bool]:
    print("=== equivalence partitioning ===")
    results: list[bool] = []

    with sandbox():
        # Output mode: verbose shows the arguments, normal mode only the name
        result: CallResult = call("get_files_info", '{"directory": "pkg"}')
        results.append(check("normal mode: name only", result.output == announced("get_files_info")))
        result = call("get_files_info", '{"directory": "pkg"}', verbose=True)
        results.append(check("verbose mode: name and arguments",
                             result.output == " - Calling function: get_files_info({'directory': 'pkg'})\n"))

        # Arguments: none, one, several (including a list value)
        result = call("get_files_info", "{}")
        results.append(check("no arguments: function defaults used",
                             result.content == get_files_info(WORKING_DIR)))
        result = call("get_file_content", '{"file_path": "notes.txt"}')
        results.append(check("one argument: passed through", result.content == "some notes\n"))
        result = call("run_python_file", '{"file_path": "argv.py", "args": ["a", "b c"]}')
        results.append(check("several arguments, list included: passed through",
                             result.content == "STDOUT:\n['a', 'b c']\n"))

        # Result kind: success and error text both come back unchanged as content
        result = call("get_file_content", '{"file_path": "missing.txt"}')
        results.append(check("error result: returned as content, not raised",
                             result.error is None
                             and result.content == get_file_content(WORKING_DIR, "missing.txt")
                             and str(result.content).startswith("Error:")))

        # Every offered function reaches the real function of the same name
        for name, arguments in [("get_files_info", '{"directory": "pkg"}'),
                                ("get_file_content", '{"file_path": "notes.txt"}'),
                                ("run_python_file", '{"file_path": "main.py"}'),
                                ("write_file", '{"file_path": "out.txt", "content": "abc"}')]:
            result = call(name, arguments)
            expected: str = function_map[name](working_directory=WORKING_DIR, **json.loads(arguments))
            results.append(check(f"{name}: dispatched to the real {name}()", result.content == expected))

        # Message shape: a tool message answering this exact call
        result = call("get_files_info", "{}", call_id="call_xyz")
        results.append(check("message: exactly role, tool_call_id and content",
                             result.message is not None
                             and set(result.message) == {"role", "tool_call_id", "content"}))
        results.append(check("message: role is 'tool' and the id echoes the call",
                             result.message is not None and result.message["role"] == "tool"
                             and result.message["tool_call_id"] == "call_xyz"))

    return results


def test_boundaries() -> list[bool]:
    print("=== boundary values ===")
    results: list[bool] = []

    with sandbox():
        listing: str = get_files_info(WORKING_DIR, ".")

        # Arguments string: null, empty, and an empty object all mean "no arguments"
        for label, arguments in [("null", None), ('""', ""), ("{}", "{}")]:
            result: CallResult = call("get_files_info", arguments)
            results.append(check(f"arguments {label} -> treated as {{}}", result.content == listing))

        result = call("get_files_info", '{"directory": ""}')
        results.append(check('directory "" -> same as "."', result.content == listing))

        # Empty results: call_function returns them as is (main.py is the one that refuses them)
        result = call("get_files_info", '{"directory": "empty_dir"}')
        results.append(check("empty directory -> content is ''", result.error is None and result.content == ""))
        result = call("get_file_content", '{"file_path": "empty.txt"}')
        results.append(check("empty file -> content is ''", result.error is None and result.content == ""))

        # File length around MAX_CHARS: content passes through, truncation note included
        make_fixture("calculator/exact.txt", "a" * MAX_CHARS)
        make_fixture("calculator/over.txt", "a" * (MAX_CHARS + 1))
        result = call("get_file_content", '{"file_path": "exact.txt"}')
        results.append(check(f"{MAX_CHARS} chars -> all of it, no truncation note",
                             result.content == "a" * MAX_CHARS))
        result = call("get_file_content", '{"file_path": "over.txt"}')
        results.append(check(f"{MAX_CHARS + 1} chars -> cut at {MAX_CHARS} with the note",
                             result.content == "a" * MAX_CHARS
                             + f'[...File "over.txt" truncated at {MAX_CHARS} characters]'))

        # Script arguments: none, an empty list, one
        for label, arguments, expected in [
            ("no args key", '{"file_path": "argv.py"}', "STDOUT:\n[]\n"),
            ("args []", '{"file_path": "argv.py", "args": []}', "STDOUT:\n[]\n"),
            ("args [x]", '{"file_path": "argv.py", "args": ["x"]}', "STDOUT:\n['x']\n"),
        ]:
            result = call("run_python_file", arguments)
            results.append(check(f"run_python_file, {label} -> {expected!r}", result.content == expected))

    return results


def test_decision_table() -> list[bool]:
    # | Rule | verbose | arguments  | announcement                       | content        |
    # |------|---------|------------|------------------------------------|----------------|
    # | R1   | N       | none       | name only                          | real result    |
    # | R2   | N       | some       | name only                          | real result    |
    # | R3   | Y       | none       | name({})                           | real result    |
    # | R4   | Y       | some       | name({...}) with the model's args  | real result    |
    print("=== decision table ===")
    results: list[bool] = []

    with sandbox():
        # (rule, verbose, arguments, expected announcement)
        rules: list[tuple[str, bool, str, str]] = [
            ("R1", False, "{}", announced("get_files_info")),
            ("R2", False, '{"directory": "pkg"}', announced("get_files_info")),
            ("R3", True, "{}", " - Calling function: get_files_info({})\n"),
            ("R4", True, '{"directory": "pkg"}', " - Calling function: get_files_info({'directory': 'pkg'})\n"),
        ]
        for rule, verbose, arguments, expected in rules:
            result: CallResult = call("get_files_info", arguments, verbose)
            directory: str = json.loads(arguments).get("directory", ".")
            results.append(check(f"{rule} verbose={verbose}, args {arguments}: announcement and result",
                                 result.output == expected
                                 and result.content == get_files_info(WORKING_DIR, directory)))

    return results


def test_working_directory() -> list[bool]:
    # The model must never get outside calculator/, and must never pick the working directory
    print("=== working directory and guardrails ===")
    results: list[bool] = []

    with sandbox() as tmp_root:
        result: CallResult = call("get_files_info", '{"working_directory": "..", "directory": "."}')
        results.append(check("model sends working_directory '..': ignored, still lists calculator/",
                             result.content == get_files_info(WORKING_DIR, ".")
                             and "secret.txt" not in str(result.content)))

        result = call("get_files_info", '{"directory": "."}', verbose=True)
        results.append(check("verbose shows the model's arguments, not the injected working_directory",
                             "working_directory" not in result.output))

        result = call("get_file_content", '{"file_path": "../secret.txt"}')
        results.append(check("read ../secret.txt: refused, secret not returned",
                             str(result.content).startswith('Error: Cannot read "../secret.txt"')
                             and "top secret" not in str(result.content)))

        secret_path: str = os.path.join(tmp_root, "secret.txt")
        result = call("get_file_content", json.dumps({"file_path": secret_path}))
        results.append(check("read the secret by absolute path: refused",
                             str(result.content).startswith("Error: Cannot read")))

        result = call("write_file", '{"file_path": "../evil.txt", "content": "pwned"}')
        results.append(check("write ../evil.txt: refused, nothing written",
                             str(result.content).startswith('Error: Cannot write to "../evil.txt"')
                             and not os.path.exists(os.path.join(tmp_root, "evil.txt"))))

        marker: str = os.path.join(tmp_root, "ran.marker")
        make_fixture(os.path.join(tmp_root, "outside.py"), f"open({marker!r}, 'w').close()\n")
        result = call("run_python_file", '{"file_path": "../outside.py"}')
        results.append(check("run ../outside.py: refused, script never ran",
                             str(result.content).startswith('Error: Cannot execute "../outside.py"')
                             and not os.path.exists(marker)))

    # What the model is offered
    names: list[str] = [tool["function"]["name"] for tool in available_functions]
    results.append(check(f"offered functions: {names}", names == EXPECTED_FUNCTIONS))
    results.append(check("every offered function can be called, and nothing else can",
                         set(function_map) == set(names)))
    results.append(check("every offered tool is a function tool",
                         all(tool["type"] == "function" for tool in available_functions)))
    for tool in available_functions:
        name: str = tool["function"]["name"]
        func: Callable[..., str] = function_map[name]
        params: set[str] = set(inspect.signature(func).parameters)
        properties: set[str] = set(schema_properties(dict(tool)))
        results.append(check(f"{name}: every schema argument is a real parameter", properties <= params))
        results.append(check(f"{name}: working_directory is not exposed to the model",
                             "working_directory" not in properties))

    return results


def test_error_guessing() -> list[bool]:
    print("=== error guessing ===")
    results: list[bool] = []

    with sandbox() as tmp_root:
        # Function-level problems come back as error text the model can read
        result: CallResult = call("get_files_info", '{"directory": 5}')
        results.append(check("directory is a number: error text, no crash",
                             result.error is None and str(result.content).startswith("Error:")))

        result = call("run_python_file", '{"file_path": "crash.py"}')
        results.append(check("script that crashes: exit code and stderr returned",
                             result.content == "Process exited with code 1\nSTDERR:\nboom\n"))

        result = call("get_files_info", '{"directory": "dossier été"}')
        results.append(check("unicode directory name: listed",
                             result.content == "- fichier.txt: file_size=8 bytes, is_dir=False"))

        # Not handled yet: these raise instead of telling the model what went wrong
        result = call("delete_all_files", "{}")
        results.append(check("made-up function: announced, then KeyError (not handled yet)",
                             result.output == announced("delete_all_files") and isinstance(result.error, KeyError)))

        marker: str = os.path.join(tmp_root, "shell.marker")
        result = call(f"get_files_info; touch {marker}", "{}")
        results.append(check("name with shell syntax: KeyError, nothing run",
                             isinstance(result.error, KeyError) and not os.path.exists(marker)))

        result = call("get_files_info", '{"directory": ')
        results.append(check("malformed JSON: JSONDecodeError before anything is announced (not handled yet)",
                             isinstance(result.error, json.JSONDecodeError) and result.output == ""))

        for arguments in ("[]", "null"):
            result = call("get_files_info", arguments)
            results.append(check(f"arguments {arguments} (valid JSON, not an object): TypeError (not handled yet)",
                                 isinstance(result.error, TypeError)))

        result = call("get_files_info", '{"file_path": "main.py"}')
        results.append(check("argument the function doesn't take: TypeError (not handled yet)",
                             isinstance(result.error, TypeError)))

        result = call("get_file_content", "{}")
        results.append(check("required argument missing: TypeError (not handled yet)",
                             isinstance(result.error, TypeError)))

    return results


def main() -> None:
    results: list[bool] = []
    results += test_course_cases()
    results += test_equivalence_partitions()
    results += test_boundaries()
    results += test_decision_table()
    results += test_working_directory()
    results += test_error_guessing()
    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
