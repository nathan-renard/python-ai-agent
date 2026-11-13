import os
import subprocess

from openai.types.chat import ChatCompletionToolParam

from config import TIMEOUT_SECONDS


def run_python_file(
    working_directory: str, file_path: str, args: list[str] | None = None
) -> str:
    try:
        # realpath resolves symlinks, so a link inside can't lead outside the working directory
        working_dir_abs: str = os.path.realpath(working_directory)
        absolute_file_path: str = os.path.realpath(os.path.join(working_dir_abs, file_path))

        # Guardrail: the LLM must never reach outside the working directory
        if os.path.commonpath([working_dir_abs, absolute_file_path]) != working_dir_abs:
            return f'Error: Cannot execute "{file_path}" as it is outside the permitted working directory'

        if not os.path.isfile(absolute_file_path):
            return f'Error: "{file_path}" does not exist or is not a regular file'

        # Check the resolved path: a "script.py" symlink to a .txt file must not run
        if not absolute_file_path.endswith(".py"):
            return f'Error: "{file_path}" is not a Python file'

        command: list[str] = ["python", absolute_file_path]
        if args:
            command.extend(args)

        # A list (no shell=True) means args are passed literally, never parsed by a shell
        result: subprocess.CompletedProcess[str] = subprocess.run(
            command,
            cwd=working_dir_abs,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )

        output: list[str] = []
        if result.returncode != 0:
            output.append(f"Process exited with code {result.returncode}")
        if not result.stdout and not result.stderr:
            output.append("No output produced")
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        return "\n".join(output)

    except Exception as e:
        return f"Error: executing Python file: {e}"


schema_run_python_file: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "run_python_file",
        "description": "Executes a Python file relative to the working directory with optional command-line arguments, returning its exit code, stdout and stderr",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the Python (.py) file to execute, relative to the working directory",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional command-line arguments to pass to the Python file",
                },
            },
            "required": ["file_path"],
        },
    },
}
