import importlib
import inspect
import io
import json
import os
import sys
import tempfile
from collections.abc import Callable
from contextlib import redirect_stdout
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCallUnion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
    Custom,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

from call_function import available_functions
from main import main as run_agent
from prompts import system_prompt


def check(label: str, passed: bool) -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}: {label}")
    return passed


def fn_call(name: str, arguments: str | None) -> ChatCompletionMessageFunctionToolCall:
    # model_construct skips validation, so a test can send arguments=None like a sloppy provider might
    return ChatCompletionMessageFunctionToolCall(
        id=f"call_{name}",
        type="function",
        function=Function.model_construct(name=name, arguments=arguments),
    )


def custom_call(name: str) -> ChatCompletionMessageCustomToolCall:
    # A non-function tool call: the API allows them, but we never offer custom tools
    return ChatCompletionMessageCustomToolCall(
        id=f"custom_{name}", type="custom", custom=Custom(name=name, input="x"))


def make_response(content: str | None,
                  tool_calls: list[ChatCompletionMessageToolCallUnion] | None = None) -> ChatCompletion:
    message: ChatCompletionMessage = ChatCompletionMessage(
        role="assistant", content=content, tool_calls=tool_calls)
    return ChatCompletion(
        id="resp_test",
        object="chat.completion",
        created=0,
        model="test-model",
        choices=[Choice(index=0, finish_reason="tool_calls" if tool_calls else "stop", message=message)],
        usage=CompletionUsage(prompt_tokens=12, completion_tokens=3, total_tokens=15),
    )


def run_main(response: ChatCompletion, argv: list[str]) -> tuple[str, dict[str, Any]]:
    # Runs main.py end to end with a fake model; returns what it printed and what it sent to the model
    fake_client: MagicMock = MagicMock()
    fake_client.chat.completions.create.return_value = response
    out: io.StringIO = io.StringIO()
    with (patch.object(sys, "argv", ["main.py", *argv]),
          patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
          patch("main.OpenAI", return_value=fake_client),
          redirect_stdout(out)):
        run_agent()
    return out.getvalue(), fake_client.chat.completions.create.call_args.kwargs


def output_for(response: ChatCompletion) -> str:
    return run_main(response, ["test prompt"])[0]


def resolve_function(name: str) -> Callable[..., object] | None:
    # Looks for functions/<name>.py with a function of the same name inside it
    try:
        module: ModuleType = importlib.import_module(f"functions.{name}")
    except ModuleNotFoundError:
        return None
    func: object = getattr(module, name, None)
    return func if callable(func) else None


def schema_properties(tool: dict[str, Any]) -> dict[str, Any]:
    return tool["function"].get("parameters", {}).get("properties", {})


def test_course_cases() -> list[bool]:
    # Replies shaped like the ones the real model sent for the course prompts
    print("=== use cases: course prompts ===")
    results: list[bool] = []

    # (prompt, arguments the model sent, expected output)
    cases: list[tuple[str, str, str]] = [
        ("what files are in the root?", '{"directory": "."}',
         "Calling function: get_files_info({'directory': '.'})\n"),
        ("what files are in the root?", "{}",
         "Calling function: get_files_info({})\n"),
        ("what files are in the pkg directory?", '{"directory": "pkg"}',
         "Calling function: get_files_info({'directory': 'pkg'})\n"),
    ]
    for prompt, arguments, expected in cases:
        output, _ = run_main(make_response(None, [fn_call("get_files_info", arguments)]), [prompt])
        print(output, end="")
        results.append(check(f"{prompt!r}, args {arguments}: announces the call", output == expected))

    output, _ = run_main(make_response("2 + 2 = 4"), ["what is 2 + 2?"])
    print(output, end="")
    results.append(check("'what is 2 + 2?', no call: prints the answer as normal", output == "2 + 2 = 4\n"))

    return results


def test_equivalence_partitions() -> list[bool]:
    print("=== equivalence partitioning ===")
    results: list[bool] = []

    # Reply shape: text only, a call only, a call plus text
    output: str = output_for(make_response("Just text"))
    results.append(check("reply: text only -> text printed", output == "Just text\n"))

    output = output_for(make_response(None, [fn_call("get_files_info", '{"directory": "pkg"}')]))
    results.append(check("reply: one call -> call announced",
                         output == "Calling function: get_files_info({'directory': 'pkg'})\n"))

    # Models often say "Let me check..." alongside the call; the text must not be printed
    output = output_for(make_response("Let me check.", [fn_call("get_files_info", "{}")]))
    results.append(check("reply: call plus text -> only the call announced",
                         output == "Calling function: get_files_info({})\n"))

    # Arguments: none, one, several (including a list value)
    # (label, arguments the model sent, expected output)
    arg_cases: list[tuple[str, str, str]] = [
        ("no arguments", "{}", "Calling function: get_files_info({})\n"),
        ("one argument", '{"directory": "pkg"}',
         "Calling function: get_files_info({'directory': 'pkg'})\n"),
        ("several arguments", '{"file_path": "main.py", "args": ["3 + 5"]}',
         "Calling function: get_files_info({'file_path': 'main.py', 'args': ['3 + 5']})\n"),
    ]
    for label, arguments, expected in arg_cases:
        output = output_for(make_response(None, [fn_call("get_files_info", arguments)]))
        results.append(check(f"arguments: {label}", output == expected))

    # What main.py sends to the model
    _, sent = run_main(make_response("ok"), ["what files are in the root?"])
    results.append(check("request: the declared tools are offered", sent["tools"] == available_functions))
    results.append(check("request: system prompt first, then the user prompt",
                         sent["messages"] == [
                             {"role": "system", "content": system_prompt},
                             {"role": "user", "content": "what files are in the root?"},
                         ]))

    # --verbose: token counts come first, then the call
    output, _ = run_main(make_response(None, [fn_call("get_files_info", "{}")]), ["list files", "--verbose"])
    results.append(check("--verbose with a call: prompt and token counts, then the call",
                         output == "User prompt: list files\n"
                                   "Prompt tokens: 12\n"
                                   "Response tokens: 3\n"
                                   "Calling function: get_files_info({})\n"))

    return results


def test_boundaries() -> list[bool]:
    print("=== boundary values ===")
    results: list[bool] = []

    # Number of tool calls: None, 0, 1, 2
    output: str = output_for(make_response("text", None))
    results.append(check("tool_calls is None -> text printed", output == "text\n"))

    output = output_for(make_response("text", []))
    results.append(check("tool_calls is [] (0 calls) -> treated as no calls, text printed",
                         output == "text\n"))

    output = output_for(make_response(None, [fn_call("get_files_info", "{}")]))
    results.append(check("1 call -> exactly 1 line", output == "Calling function: get_files_info({})\n"))

    output = output_for(make_response(None, [
        fn_call("get_files_info", '{"directory": "."}'),
        fn_call("get_files_info", '{"directory": "pkg"}'),
    ]))
    results.append(check("2 calls -> 2 lines, in the order the model sent them",
                         output == "Calling function: get_files_info({'directory': '.'})\n"
                                   "Calling function: get_files_info({'directory': 'pkg'})\n"))

    # Arguments string: empty string vs an empty JSON object
    output = output_for(make_response(None, [fn_call("get_files_info", "")]))
    results.append(check('arguments "" -> treated as {}', output == "Calling function: get_files_info({})\n"))

    output = output_for(make_response(None, [fn_call("get_files_info", '{"directory": ""}')]))
    results.append(check("empty directory value -> passed through as ''",
                         output == "Calling function: get_files_info({'directory': ''})\n"))

    # Content: empty string with no calls
    output = output_for(make_response(""))
    results.append(check('content "" -> one blank line', output == "\n"))

    return results


def test_decision_table() -> list[bool]:
    # | Rule | function calls | custom calls | content | expected                                     |
    # |------|----------------|--------------|---------|----------------------------------------------|
    # | R1   | 0              | 0            | Y       | content printed                              |
    # | R2   | 0              | 0            | N       | "None" printed (no guard for this yet)       |
    # | R3   | >=1            | 0            | N       | one line per call                            |
    # | R4   | >=1            | 0            | Y       | calls printed, content not printed           |
    # | R5   | 0              | >=1          | Y       | nothing printed, custom call skipped         |
    # | R6   | >=1            | >=1          | N       | only the function calls printed              |
    print("=== decision table ===")
    results: list[bool] = []
    files_call: str = "Calling function: get_files_info({})\n"

    output: str = output_for(make_response("hello"))
    results.append(check("R1 no calls, content: content printed", output == "hello\n"))

    output = output_for(make_response(None))
    results.append(check("R2 no calls, no content: prints 'None'", output == "None\n"))

    output = output_for(make_response(None, [fn_call("get_files_info", "{}")]))
    results.append(check("R3 function call, no content: call announced", output == files_call))

    output = output_for(make_response("Let me look.", [fn_call("get_files_info", "{}")]))
    results.append(check("R4 function call, content: only the call announced", output == files_call))

    output = output_for(make_response("Let me look.", [custom_call("grammar_tool")]))
    results.append(check("R5 custom call only: nothing printed, no crash", output == ""))

    output = output_for(make_response(None, [custom_call("grammar_tool"), fn_call("get_files_info", "{}")]))
    results.append(check("R6 custom + function call: only the function call announced",
                         output == files_call))

    return results


def test_unknown_functions() -> list[bool]:
    # The model is only offered functions that exist, and at this step a call is
    # only announced: nothing gets run, whether the function exists or not
    print("=== unknown and unoffered functions ===")
    results: list[bool] = []

    names: list[str] = [tool["function"]["name"] for tool in available_functions]
    results.append(check(f"offered functions: {names}", names == ["get_files_info"]))
    results.append(check("every offered tool is a function tool",
                         all(tool["type"] == "function" for tool in available_functions)))
    results.append(check("no function is offered twice", len(names) == len(set(names))))

    for tool in available_functions:
        name: str = tool["function"]["name"]
        func: Callable[..., object] | None = resolve_function(name)
        results.append(check(f"{name}: exists as functions/{name}.py:{name}()", func is not None))
        if func is None:
            continue
        params: set[str] = set(inspect.signature(func).parameters)
        properties: set[str] = set(schema_properties(dict(tool)))
        results.append(check(f"{name}: every schema argument is a real parameter",
                             properties <= params))
        # working_directory is injected by us, so the model must never be able to pick it
        results.append(check(f"{name}: working_directory is not exposed to the model",
                             "working_directory" not in properties))

    # A made-up function: announced, no crash, nothing else printed
    output: str = output_for(make_response(None, [fn_call("delete_all_files", '{"path": "/"}')]))
    results.append(check("made-up function: only announced",
                         output == "Calling function: delete_all_files({'path': '/'})\n"))

    # A name with shell syntax in it: printed as text, never run
    output = output_for(make_response(None, [fn_call("get_files_info; rm -rf /", "{}")]))
    results.append(check("name with shell syntax: printed verbatim",
                         output == "Calling function: get_files_info; rm -rf /({})\n"))

    with tempfile.TemporaryDirectory() as tmp_root:
        # Real functions that aren't offered yet: a call to them must not touch the disk
        target: str = os.path.join(tmp_root, "written.txt")
        args: str = json.dumps({"file_path": target, "content": "pwned"})
        output = output_for(make_response(None, [fn_call("write_file", args)]))
        results.append(check("write_file (not offered): announced only, no file written",
                             output.count("\n") == 1 and not os.path.exists(target)))

        marker: str = os.path.join(tmp_root, "ran.marker")
        script: str = os.path.join(tmp_root, "script.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(f"open({marker!r}, 'w').close()\n")
        args = json.dumps({"file_path": script})
        output = output_for(make_response(None, [fn_call("run_python_file", args)]))
        results.append(check("run_python_file (not offered): announced only, script never ran",
                             output.count("\n") == 1 and not os.path.exists(marker)))

    # An offered function isn't run either: no file listing appears after the announcement
    output = output_for(make_response(None, [fn_call("get_files_info", '{"directory": "."}')]))
    results.append(check("get_files_info: announced only, no listing printed",
                         output == "Calling function: get_files_info({'directory': '.'})\n"))

    return results


def test_error_guessing() -> list[bool]:
    print("=== error guessing ===")
    results: list[bool] = []

    # Some providers send null instead of "" when there are no arguments
    output: str = output_for(make_response(None, [fn_call("get_files_info", None)]))
    results.append(check("arguments null -> treated as {}", output == "Calling function: get_files_info({})\n"))

    # Non-ASCII and spaces survive the JSON round trip
    output = output_for(make_response(None, [fn_call("get_files_info", '{"directory": "dossier été"}')]))
    results.append(check("unicode directory name printed intact",
                         output == "Calling function: get_files_info({'directory': 'dossier été'})\n"))

    # Truncated JSON from the model: json.loads isn't guarded yet, so main.py stops with an exception
    try:
        output_for(make_response(None, [fn_call("get_files_info", '{"directory": ')]))
        raised: bool = False
    except json.JSONDecodeError:
        raised = True
    results.append(check("malformed JSON arguments: raises JSONDecodeError (not handled yet)", raised))

    return results


def main() -> None:
    results: list[bool] = []
    results += test_course_cases()
    results += test_equivalence_partitions()
    results += test_boundaries()
    results += test_decision_table()
    results += test_unknown_functions()
    results += test_error_guessing()
    print(f"\n{sum(results)}/{len(results)} checks passed")


if __name__ == "__main__":
    main()
