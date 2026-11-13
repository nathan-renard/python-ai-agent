import os

from openai.types.chat import ChatCompletionToolParam

from config import MAX_CHARS


def get_file_content(working_directory: str, file_path: str) -> str:
    try:
        # realpath resolves symlinks, so a link inside can't lead outside the working directory
        working_dir_abs: str = os.path.realpath(working_directory)
        target_file: str = os.path.realpath(os.path.join(working_dir_abs, file_path))

        # Guardrail: the LLM must never reach outside the working directory
        if os.path.commonpath([working_dir_abs, target_file]) != working_dir_abs:
            return f'Error: Cannot read "{file_path}" as it is outside the permitted working directory'

        if not os.path.isfile(target_file):
            return f'Error: File not found or is not a regular file: "{file_path}"'

        with open(target_file, "r") as f:
            content: str = f.read(MAX_CHARS)
            # If anything is left after MAX_CHARS, the file was cut off
            if f.read(1):
                content += f'[...File "{file_path}" truncated at {MAX_CHARS} characters]'
        return content

    except Exception as e:
        return f"Error: {e}"


schema_get_file_content: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_file_content",
        "description": f"Reads and returns the contents of a file relative to the working directory, truncated at {MAX_CHARS} characters",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the file to read, relative to the working directory",
                },
            },
            "required": ["file_path"],
        },
    },
}
