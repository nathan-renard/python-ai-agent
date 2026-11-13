import os

from openai.types.chat import ChatCompletionToolParam


def get_files_info(working_directory: str, directory: str = ".") -> str:
    try:
        # realpath resolves symlinks, so a link inside can't lead outside the working directory
        working_dir_abs: str = os.path.realpath(working_directory)
        target_dir: str = os.path.realpath(os.path.join(working_dir_abs, directory))

        # Guardrail: the LLM must never reach outside the working directory
        if os.path.commonpath([working_dir_abs, target_dir]) != working_dir_abs:
            return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'

        if not os.path.isdir(target_dir):
            return f'Error: "{directory}" is not a directory'

        entries: list[str] = []
        for name in sorted(os.listdir(target_dir)):
            path: str = os.path.join(target_dir, name)
            file_size: int = os.path.getsize(path)
            is_dir: bool = os.path.isdir(path)
            entries.append(f"- {name}: file_size={file_size} bytes, is_dir={is_dir}")
        return "\n".join(entries)
    
    except Exception as e:
        return f"Error: {e}"


schema_get_files_info: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_files_info",
        "description": "Lists files in a specified directory relative to the working directory, providing file size and directory status",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list files from, relative to the working directory (default is the working directory itself)",
                },
            },
        },
    },
}
