import os
from config import MAX_CHARS
from google.genai import types

def get_file_content(working_directory, file_path):
    abs_working_directory = os.path.abspath(working_directory)
    target_file = os.path.abspath(os.path.join(working_directory, file_path))
    print(target_file)
    if not target_file.startswith(abs_working_directory):
        return f'Error: Cannot read "{file_path}" as it is outside the permitted working directory'
    if not os.path.isfile(target_file):
        return f'Error: File not found or is not a regular file: "{file_path}"'

    try:
        with open(target_file, 'r') as f:
            file_content_string = f.read(MAX_CHARS)
            if len(file_content_string) >= MAX_CHARS:
                file_content_string += f'...File "{file_path}" truncated at 10000 characters'
        print(file_content_string)
            
    except Exception as e:
        return f'Error: {e}'
    

schema_get_file_content = types.FunctionDeclaration(
    name="get_file_content",
    description=f"Reads the first {MAX_CHARS} characters of a file in the specified working directory.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "file_path": types.Schema(
                type=types.Type.STRING,
                description="The path to the file to read, relative to the working directory.",
            ),
        },
    ),
)