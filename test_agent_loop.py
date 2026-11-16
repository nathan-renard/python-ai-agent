import copy
import io
import json
import os
import sys
import tempfile
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionToolMessageParam,
)
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
from config import MAX_ITERATIONS
from functions.get_file_content import get_file_content
from functions.get_files_info import get_files_info
from functions.run_python_file import run_python_file
from main import main as run_main
from prompts import system_prompt

# Given the 0-based number of the model call, returns what the model replies
type Reply = Callable[[int], ChatCompletion]

# call_function.py injects this, so it is the oracle for what the real functions return
WORKING_DIR: str = "./calculator"
LIMIT_ERROR: str = f"Error: reached the maximum of {MAX_ITERATIONS} iterations without a final response\n"
USAGE: CompletionUsage = CompletionUsage(prompt_tokens=12, completion_tokens=3, total_tokens=15)


@dataclass
class AgentRun:
    output: str                      # everything main.py printed
    exit_code: int | str | None      # None means main() returned normally
    error: Exception | None          # an exception main.py let escape
    sent: list[list[Any]]            # the messages, as they were at each model call
    tools: list[Any]                 # the tools offered at each model call

    @property
    def calls(self) -> int:
        return len(self.sent)

    # These return []/{} instead of raising IndexError, so a missing message shows up as a FAIL line, not a crash
    def messages(self, call: int) -> list[Any]:
        return self.sent[call] if -len(self.sent) <= call < len(self.sent) else []

    def message(self, call: int, index: int) -> dict[str, Any]:
        found: list[Any] = self.messages(call)
        return found[index] if -len(found) <= index < len(found) else {}


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


def custom_call(name: str, call_id: str = "custom_1") -> ChatCompletionMessageCustomToolCall:
    # A non-function tool call: the API allows them, but we never offer custom tools
    return ChatCompletionMessageCustomToolCall(id=call_id, type="custom", custom=Custom(name=name, input="x"))


def make_response(content: str | None,
                  tool_calls: list[ChatCompletionMessageToolCallUnion] | None = None,
                  usage: CompletionUsage | None = USAGE) -> ChatCompletion:
    message: ChatCompletionMessage = ChatCompletionMessage(
        role="assistant", content=content, tool_calls=tool_calls)
    return ChatCompletion(
        id="resp_test",
        object="chat.completion",
        created=0,
        model="test-model",
        choices=[Choice(index=0, finish_reason="tool_calls" if tool_calls else "stop", message=message)],
        usage=usage,
    )


def calls_then_answer(tool_rounds: int, answer: str = "done") -> Reply:
    # get_files_info once per round for tool_rounds rounds, then a final answer
    def reply(i: int) -> ChatCompletion:
        if i < tool_rounds:
            return make_response(None, [fn_call("get_files_info", "{}", f"call_{i + 1}")])
        return make_response(answer)
    return reply


def script(*responses: ChatCompletion) -> Reply:
    # Replies in order; a model call past the end raises IndexError, so an extra call shows up as an error
    return lambda i: responses[i]


def always(response: ChatCompletion) -> Reply:
    # A model that never stops asking
    return lambda _: response


def run_agent(reply: Reply, argv: list[str] | None = None) -> AgentRun:
    # Runs main.py end to end with a fake model; the functions it calls are the real ones
    sent: list[list[Any]] = []
    tools: list[Any] = []

    def fake_create(**kwargs: Any) -> ChatCompletion:
        # deepcopy: main.py keeps appending to the same list, so a plain reference would show later messages too
        sent.append(copy.deepcopy(kwargs["messages"]))
        tools.append(kwargs["tools"])
        return reply(len(sent) - 1)

    fake_client: MagicMock = MagicMock()
    fake_client.chat.completions.create.side_effect = fake_create
    out: io.StringIO = io.StringIO()
    exit_code: int | str | None = None
    error: Exception | None = None
    with (patch.object(sys, "argv", ["main.py", *(argv or ["test prompt"])]),
          patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
          patch("main.OpenAI", return_value=fake_client),
          redirect_stdout(out)):
        try:
            run_main()
        except SystemExit as e:
            exit_code = e.code
        except Exception as e:
            error = e
    return AgentRun(out.getvalue(), exit_code, error, sent, tools)


def tool_message(call_id: str, content: str) -> ChatCompletionToolMessageParam:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def is_prefix(shorter: list[Any], longer: list[Any]) -> bool:
    return longer[:len(shorter)] == shorter


def test_course_cases() -> list[bool]:
    print("=== use cases: course prompts ===")
    results: list[bool] = []

    # The example from the lesson: list the files, read render.py, then answer
    answer: str = ("The calculator evaluates the expression, formats the result (along with the\n"
                   "original expression) into a JSON-like string, and then prints that string to\n"
                   "the console.")
    run: AgentRun = run_agent(script(
        make_response(None, [fn_call("get_files_info", "{}", "call_1")]),
        make_response(None, [fn_call("get_file_content", '{"file_path": "pkg/render.py"}', "call_2")]),
        make_response(answer),
    ), ["how does the calculator render results to the console?"])
    print(run.output, end="")
    results.append(check("render question: output matches the lesson example",
                         run.output == " - Calling function: get_files_info\n"
                                       " - Calling function: get_file_content\n"
                                       f"Final response:\n{answer}\n"))
    results.append(check("render question: 3 model calls, then main() returns normally",
                         run.calls == 3 and run.exit_code is None and run.error is None))
    results.append(check("render question: the last call saw the real contents of pkg/render.py",
                         run.message(2, -1) == tool_message("call_2", get_file_content(WORKING_DIR, "pkg/render.py"))))

    # Running a script: the model sees the real program output
    run = run_agent(script(
        make_response(None, [fn_call("run_python_file", '{"file_path": "main.py", "args": ["3 + 5"]}')]),
        make_response("3 + 5 is 8"),
    ), ["what is 3 + 5 according to the calculator?"])
    print(run.output, end="")
    results.append(check("run calculator: call announced, then the final response",
                         run.output == " - Calling function: run_python_file\nFinal response:\n3 + 5 is 8\n"))
    results.append(check("run calculator: the model saw the script's real output",
                         run.message(1, -1) == tool_message("call_1", run_python_file(WORKING_DIR, "main.py", ["3 + 5"]))))

    # No tools needed: answered on the first call
    run = run_agent(script(make_response("2 + 2 = 4")), ["what is 2 + 2?"])
    print(run.output, end="")
    results.append(check("'what is 2 + 2?': answered in 1 call",
                         run.output == "Final response:\n2 + 2 = 4\n" and run.calls == 1))

    return results


def test_message_history() -> list[bool]:
    # Every call must see everything said so far, including the function results
    print("=== message history ===")
    results: list[bool] = []

    run: AgentRun = run_agent(script(
        make_response(None, [fn_call("get_files_info", '{"directory": "pkg"}', "call_1")]),
        make_response("pkg holds the calculator code"),
    ), ["what is in pkg?"])
    start: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "what is in pkg?"},
    ]
    results.append(check("call 1: system prompt, then the user prompt", run.messages(0) == start))
    results.append(check("call 2: starts with everything call 1 saw", is_prefix(start, run.messages(1))))
    results.append(check("call 2: the model's tool call request comes next, as sent",
                         run.message(1, 2) == {
                             "role": "assistant",
                             "tool_calls": [{"id": "call_1", "type": "function",
                                             "function": {"name": "get_files_info",
                                                          "arguments": '{"directory": "pkg"}'}}],
                         }))
    results.append(check("call 2: then the tool message: role tool, matching id, real listing as content",
                         run.message(1, 3) == tool_message("call_1", get_files_info(WORKING_DIR, "pkg"))))
    results.append(check("call 2: nothing else was added", len(run.messages(1)) == 4))
    results.append(check("every call is offered the tools", run.tools == [available_functions] * 2))

    # Text the model sent alongside a call stays in the history
    run = run_agent(script(
        make_response("Let me check.", [fn_call("get_files_info", "{}")]),
        make_response("done"),
    ))
    results.append(check("text sent with a call is kept in the history",
                         run.message(1, 2).get("content") == "Let me check."))

    # Two calls in one reply: two tool messages, in the order the model asked
    run = run_agent(script(
        make_response(None, [
            fn_call("get_files_info", '{"directory": "."}', "call_a"),
            fn_call("get_files_info", '{"directory": "pkg"}', "call_b"),
        ]),
        make_response("done"),
    ))
    results.append(check("2 calls in one reply -> 2 tool messages, in order, each with its own id",
                         run.messages(1)[3:] == [
                             tool_message("call_a", get_files_info(WORKING_DIR, ".")),
                             tool_message("call_b", get_files_info(WORKING_DIR, "pkg")),
                         ]))

    # Over many rounds the history only grows: each call starts with what the previous one saw
    run = run_agent(calls_then_answer(5))
    results.append(check("5 rounds: history only grows, never rewritten",
                         all(is_prefix(run.sent[i], run.sent[i + 1]) for i in range(run.calls - 1))))
    results.append(check("5 rounds: each round adds exactly 2 messages (request + result)",
                         [len(m) for m in run.sent] == [2, 4, 6, 8, 10, 12]))
    results.append(check("5 rounds: tool messages answer the matching call ids",
                         [m["tool_call_id"] for m in run.messages(-1) if m["role"] == "tool"]
                         == [f"call_{i}" for i in range(1, 6)]))

    # Through the real SDK, no fake client: this is the JSON that goes over the wire
    requests: list[dict[str, Any]] = []
    bodies: list[dict[str, Any]] = [
        make_response(None, [fn_call("get_files_info", "{}", "call_1")]).model_dump(mode="json"),
        make_response("done").model_dump(mode="json"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=bodies[len(requests) - 1])

    real_openai: type[openai.OpenAI] = openai.OpenAI

    def make_client(**kwargs: Any) -> openai.OpenAI:
        return real_openai(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    out: io.StringIO = io.StringIO()
    with (patch.object(sys, "argv", ["main.py", "list the files"]),
          patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
          patch("main.OpenAI", side_effect=make_client),
          redirect_stdout(out)):
        try:
            run_main()
        except Exception as e:
            print(f"  main.py raised {e!r}")
    wire: list[dict[str, Any]] = requests[1]["messages"] if len(requests) > 1 else []
    results.append(check("wire: 2 HTTP requests, the second carries the full history",
                         len(requests) == 2 and [m["role"] for m in wire] == ["system", "user", "assistant", "tool"]))
    results.append(check("wire: the tool message is plain JSON with the call id and real result",
                         wire[3:] == [tool_message("call_1", get_files_info(WORKING_DIR, "."))]))
    results.append(check("wire: the assistant message carries no null fields",
                         len(wire) > 2 and None not in wire[2].values()))

    return results


def test_boundaries() -> list[bool]:
    print("=== boundary values ===")
    results: list[bool] = []

    results.append(check(f"the limit is 20 (config.MAX_ITERATIONS = {MAX_ITERATIONS})", MAX_ITERATIONS == 20))

    # Rounds of tool calls before the answer: 0, 1, 18 and 19 still leave a call for the answer
    for rounds in (0, 1, MAX_ITERATIONS - 2, MAX_ITERATIONS - 1):
        run: AgentRun = run_agent(calls_then_answer(rounds))
        results.append(check(f"{rounds} rounds then an answer -> {rounds + 1} calls, answer printed, no error",
                             run.calls == rounds + 1 and run.exit_code is None
                             and run.output.endswith("Final response:\ndone\n")))

    # 20 rounds: the answer would come on call 21, which never happens
    run = run_agent(calls_then_answer(MAX_ITERATIONS))
    results.append(check("20 rounds -> stops after exactly 20 calls, exit code 1",
                         run.calls == MAX_ITERATIONS and run.exit_code == 1))
    results.append(check("20 rounds -> error is the last line, no final response",
                         run.output.endswith(LIMIT_ERROR) and "Final response:" not in run.output))
    results.append(check("20 rounds -> the 20th call saw all 40 messages",
                         len(run.messages(-1)) == 2 + 2 * (MAX_ITERATIONS - 1)))

    # A model that never stops
    run = run_agent(always(make_response(None, [fn_call("get_files_info", "{}")])))
    results.append(check("never answers -> 20 calls, 20 announcements, exit code 1",
                         run.calls == MAX_ITERATIONS and run.exit_code == 1
                         and run.output.count("Calling function") == MAX_ITERATIONS))

    # The limit itself at its edges (patched)
    with patch("main.MAX_ITERATIONS", 1):
        run = run_agent(calls_then_answer(0))
        results.append(check("limit 1, answer on call 1 -> answer printed", run.output == "Final response:\ndone\n"))
        run = run_agent(calls_then_answer(1))
        results.append(check("limit 1, tool call on call 1 -> exit code 1 after 1 call",
                             run.calls == 1 and run.exit_code == 1))
    with patch("main.MAX_ITERATIONS", 0):
        run = run_agent(calls_then_answer(0))
        results.append(check("limit 0 -> the model is never called, exit code 1",
                             run.calls == 0 and run.exit_code == 1))

    # Tool calls per reply: None and [] both mean "done"
    run = run_agent(script(make_response("text", None)))
    results.append(check("tool_calls None -> final answer", run.output == "Final response:\ntext\n"))
    run = run_agent(script(make_response("text", [])))
    results.append(check("tool_calls [] (0 calls) -> final answer, not another round",
                         run.output == "Final response:\ntext\n" and run.calls == 1))

    # Final content: empty string
    run = run_agent(script(make_response("")))
    results.append(check('final content "" -> label then a blank line', run.output == "Final response:\n\n"))

    return results


def test_decision_table() -> list[bool]:
    # | Rule | reply has                | result content | calls left | expected                               |
    # |------|--------------------------|----------------|------------|----------------------------------------|
    # | R1   | no calls, text           | -              | any        | final response printed, main() returns |
    # | R2   | no calls, no text        | -              | any        | "None" printed (no guard for this yet) |
    # | R3   | function call            | non-empty      | yes        | result appended, model called again    |
    # | R4   | function call            | non-empty      | none       | error printed, exit code 1             |
    # | R5   | function call            | empty          | any        | RuntimeError, loop stops               |
    # | R6   | custom call only         | -              | yes        | nothing run, model called again        |
    print("=== decision table ===")
    results: list[bool] = []

    run: AgentRun = run_agent(script(make_response("hello")))
    results.append(check("R1 no calls, text: printed, normal return",
                         run.output == "Final response:\nhello\n" and run.exit_code is None))

    run = run_agent(script(make_response(None)))
    results.append(check("R2 no calls, no text: prints 'None'", run.output == "Final response:\nNone\n"))

    run = run_agent(calls_then_answer(1))
    results.append(check("R3 call with a result: appended, model called again",
                         run.calls == 2 and run.message(1, -1).get("role") == "tool"))

    with patch("main.MAX_ITERATIONS", 3):
        run = run_agent(calls_then_answer(3))
    results.append(check("R4 call on the last allowed iteration: error, exit code 1",
                         run.calls == 3 and run.exit_code == 1
                         and run.output.endswith("Error: reached the maximum of 3 iterations without a final response\n")))

    with tempfile.TemporaryDirectory(dir=WORKING_DIR) as empty_dir:
        # get_files_info on an empty directory returns "", which main.py refuses
        args: str = json.dumps({"directory": os.path.basename(empty_dir)})
        run = run_agent(always(make_response(None, [fn_call("get_files_info", args)])))
    results.append(check("R5 empty result (empty directory): RuntimeError, model not called again",
                         isinstance(run.error, RuntimeError) and run.calls == 1))

    run = run_agent(script(make_response(None, [custom_call("grammar_tool")]), make_response("done")))
    results.append(check("R6 custom call only: nothing run or announced, model called again",
                         run.output == "Final response:\ndone\n" and run.calls == 2
                         and all(m["role"] != "tool" for m in run.messages(1))))

    return results


def test_verbose() -> list[bool]:
    print("=== --verbose ===")
    results: list[bool] = []

    run: AgentRun = run_agent(script(
        make_response(None, [fn_call("get_files_info", '{"directory": "pkg"}', "call_1")]),
        make_response(None, [fn_call("get_file_content", '{"file_path": "lorem.txt"}', "call_2")]),
        make_response("done"),
    ), ["look around", "--verbose"])
    tokens: str = "Prompt tokens: 12\nResponse tokens: 3\n"
    results.append(check("prompt once, tokens per call, args and results shown, then the answer",
                         run.output == "User prompt: look around\n"
                                       + tokens
                                       + " - Calling function: get_files_info({'directory': 'pkg'})\n"
                                       + f"-> {get_files_info(WORKING_DIR, 'pkg')}\n"
                                       + tokens
                                       + " - Calling function: get_file_content({'file_path': 'lorem.txt'})\n"
                                       + f"-> {get_file_content(WORKING_DIR, 'lorem.txt')}\n"
                                       + tokens
                                       + "Final response:\ndone\n"))

    run = run_agent(script(make_response("done", usage=None)), ["hi", "--verbose"])
    results.append(check("no usage in the reply -> no token lines",
                         run.output == "User prompt: hi\nFinal response:\ndone\n"))

    run = run_agent(calls_then_answer(MAX_ITERATIONS), ["hi", "--verbose"])
    results.append(check("limit reached with --verbose: still exit code 1, error last",
                         run.exit_code == 1 and run.output.endswith(LIMIT_ERROR)))

    return results


def test_error_guessing() -> list[bool]:
    print("=== error guessing ===")
    results: list[bool] = []

    # A function error is just a result: the model sees it and can recover
    run: AgentRun = run_agent(script(
        make_response(None, [fn_call("get_file_content", '{"file_path": "missing.py"}', "call_1")]),
        make_response(None, [fn_call("get_files_info", "{}", "call_2")]),
        make_response("there is no missing.py"),
    ))
    results.append(check("missing file: error goes back to the model as the tool result",
                         run.message(1, -1) == tool_message("call_1", get_file_content(WORKING_DIR, "missing.py"))
                         and str(run.message(1, -1).get("content", "")).startswith("Error:")))
    results.append(check("missing file: the model recovers and answers",
                         run.output.endswith("Final response:\nthere is no missing.py\n") and run.exit_code is None))

    run = run_agent(script(
        make_response(None, [fn_call("get_file_content", '{"file_path": "../main.py"}')]),
        make_response("not allowed"),
    ))
    results.append(check("path outside the working dir: refused, refusal sent to the model",
                         str(run.message(1, -1).get("content", "")).startswith('Error: Cannot read "../main.py"')))

    # A model stuck repeating the same failing call still stops at the limit
    run = run_agent(always(make_response(None, [fn_call("get_file_content", '{"file_path": "missing.py"}')])))
    results.append(check("same failing call forever -> stops at 20, exit code 1",
                         run.calls == MAX_ITERATIONS and run.exit_code == 1))

    # So does a model that only sends custom calls, which never produce a result
    run = run_agent(always(make_response(None, [custom_call("grammar_tool")])))
    results.append(check("custom calls forever -> stops at 20, exit code 1",
                         run.calls == MAX_ITERATIONS and run.exit_code == 1))

    # Not handled yet: these stop main.py with an exception instead of telling the model
    run = run_agent(always(make_response(None, [fn_call("delete_all_files", "{}")])))
    results.append(check("made-up function name: KeyError (not handled yet)",
                         isinstance(run.error, KeyError) and run.calls == 1))

    run = run_agent(always(make_response(None, [fn_call("get_files_info", '{"directory": ')])))
    results.append(check("malformed JSON arguments: JSONDecodeError (not handled yet)",
                         isinstance(run.error, json.JSONDecodeError) and run.calls == 1))

    with tempfile.NamedTemporaryFile(dir=WORKING_DIR, suffix=".txt") as empty_file:
        args: str = json.dumps({"file_path": os.path.basename(empty_file.name)})
        run = run_agent(always(make_response(None, [fn_call("get_file_content", args)])))
    results.append(check("reading an empty file: RuntimeError (empty result is refused)",
                         isinstance(run.error, RuntimeError)))

    def api_down(_: int) -> ChatCompletion:
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://openrouter.ai/api/v1"))

    run = run_agent(api_down)
    results.append(check("API unreachable: the error escapes, nothing printed (not handled yet)",
                         isinstance(run.error, openai.APIConnectionError) and run.output == ""))

    return results


def main() -> None:
    results: list[bool] = []
    results += test_course_cases()
    results += test_message_history()
    results += test_boundaries()
    results += test_decision_table()
    results += test_verbose()
    results += test_error_guessing()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    # A non-zero exit lets CI tell a failing run from a passing one
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
