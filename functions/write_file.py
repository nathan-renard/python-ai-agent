import os

from openai.types.chat import ChatCompletionToolParam


def write_file(working_directory: str, file_path: str, content: str) -> str:
    try:
        # realpath resolves symlinks, so a link inside can't lead outside the working directory
        working_dir_abs: str = os.path.realpath(working_directory)
        target_file: str = os.path.realpath(os.path.join(working_dir_abs, file_path))

        # Guardrail: the LLM must never reach outside the working directory
        if os.path.commonpath([working_dir_abs, target_file]) != working_dir_abs:
            return f'Error: Cannot write to "{file_path}" as it is outside the permitted working directory'

        if os.path.isdir(target_file):
            return f'Error: Cannot write to "{file_path}" as it is a directory'

        # Create any missing parent directories; no-op if they already exist
        os.makedirs(os.path.dirname(target_file), exist_ok=True)

        with open(target_file, "w") as f:
            f.write(content)
        return f'Successfully wrote to "{file_path}" ({len(content)} characters written)'

    except Exception as e:
        return f"Error: {e}"


schema_write_file: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Writes content to a file relative to the working directory, creating the file and any missing parent directories, or overwriting it if it exists",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path of the file to write, relative to the working directory",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        },
    },
}
