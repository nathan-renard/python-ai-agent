import argparse
import os
import sys
from typing import cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)

from call_function import available_functions, call_function
from config import MAX_ITERATIONS
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

    if args.verbose:
        print(f"User prompt: {args.user_prompt}")

    for _ in range(MAX_ITERATIONS):
        response: ChatCompletion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=available_functions,
        )

        if args.verbose and response.usage is not None:
            print(f"Prompt tokens: {response.usage.prompt_tokens}")
            print(f"Response tokens: {response.usage.completion_tokens}")

        message: ChatCompletionMessage = response.choices[0].message
        # Keep the model's reply in the history so the next call sees it.
        # The SDK hands back a pydantic object; model_dump turns it into the dict shape messages holds.
        messages.append(cast(ChatCompletionAssistantMessageParam, message.model_dump(exclude_none=True)))

        # No tool calls means the model is done and this is its answer
        if not message.tool_calls:
            print("Final response:")
            print(message.content)
            return

        for tool_call in message.tool_calls:
            # tool_calls can also hold custom tool calls, which have no .function; we only offer functions
            if tool_call.type != "function":
                continue
            result_message: ChatCompletionToolMessageParam = call_function(tool_call, args.verbose)
            if not result_message["content"]:
                raise RuntimeError(f"Function {tool_call.function.name} returned no content")
            if args.verbose:
                print(f"-> {result_message['content']}")
            # The model needs each result to decide its next step
            messages.append(result_message)

    print(f"Error: reached the maximum of {MAX_ITERATIONS} iterations without a final response")
    sys.exit(1)


if __name__ == "__main__":
    main()
