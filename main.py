import argparse
import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageParam

from call_function import available_functions
from prompts import system_prompt

MODEL: str = "openrouter/free"


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args: argparse.Namespace = parser.parse_args()

    load_dotenv()
    api_key: str | None = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in the environment variables.")

    client: OpenAI = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]
    response: ChatCompletion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=available_functions,
    )

    if args.verbose:
        print(f"User prompt: {args.user_prompt}")
        if response.usage is not None:
            print(f"Prompt tokens: {response.usage.prompt_tokens}")
            print(f"Response tokens: {response.usage.completion_tokens}")

    message: ChatCompletionMessage = response.choices[0].message
    if message.tool_calls:
        for tool_call in message.tool_calls:
            # tool_calls can also hold custom tool calls, which have no .function; we only offer functions
            if tool_call.type != "function":
                continue
            function_args: dict[str, Any] = json.loads(tool_call.function.arguments or "{}")
            print(f"Calling function: {tool_call.function.name}({function_args})")
    else:
        print(message.content)


if __name__ == "__main__":
    main()
