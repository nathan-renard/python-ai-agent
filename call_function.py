import json
from collections.abc import Callable
from typing import Any

from openai.types.chat import ChatCompletionMessageFunctionToolCall, ChatCompletionToolParam

from functions.get_file_content import get_file_content, schema_get_file_content
from functions.get_files_info import get_files_info, schema_get_files_info
from functions.run_python_file import run_python_file, schema_run_python_file
from functions.write_file import schema_write_file, write_file

available_functions: list[ChatCompletionToolParam] = [
    schema_get_files_info,
    schema_get_file_content,
    schema_run_python_file,
    schema_write_file,
]

function_map: dict[str, Callable[..., str]] = {
    "get_files_info": get_files_info,
    "get_file_content": get_file_content,
    "run_python_file": run_python_file,
    "write_file": write_file,
}


def call_function(tool_call: ChatCompletionMessageFunctionToolCall, verbose: bool = False) -> dict[str, str]:
    function_name: str = tool_call.function.name
    # Some providers send "" or null when there are no arguments
    function_args: dict[str, Any] = json.loads(tool_call.function.arguments or "{}")

    if verbose:
        print(f" - Calling function: {function_name}({function_args})")
    else:
        print(f" - Calling function: {function_name}")

    # The LLM doesn't know about working_directory, so we inject it ourselves
    function_args["working_directory"] = "./calculator"

    result: str = function_map[function_name](**function_args)

    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result,
    }
