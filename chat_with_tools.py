#!/usr/bin/env python3
"""
Interactive chat with gpt-oss-20b using vLLM's /v1/responses API with tools.

Features:
- Code interpreter (auto-executed by vLLM)
- Web search (if EXA_API_KEY is set)
- Custom function tools (you handle execution)
- Multi-turn conversations
- Streaming support

Usage:
    python chat_with_tools.py
    python chat_with_tools.py --no-stream
    python chat_with_tools.py --no-code  # disable code_interpreter
"""

import argparse
import json
import sys
import requests
from typing import Generator

BASE_URL = "http://localhost:8000"
MODEL = "openai/gpt-oss-20b"


def stream_response(input_text: str, tools: list, previous_id: str = None) -> Generator[dict, None, dict]:
    """Stream a response from the API, yielding events as they arrive."""
    payload = {
        "model": MODEL,
        "input": input_text,
        "tools": tools,
        "stream": True,
    }
    if previous_id:
        payload["previous_response_id"] = previous_id

    resp = requests.post(
        f"{BASE_URL}/v1/responses",
        json=payload,
        stream=True,
        headers={"Content-Type": "application/json"}
    )

    response_id = None
    full_text = ""

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode('utf-8')
        if not line.startswith('data: '):
            continue
        data = line[6:]
        if data == '[DONE]':
            break

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        event_type = event.get('type', '')

        # Capture response ID
        if event_type == 'response.created':
            response_id = event.get('response', {}).get('id')

        yield event

    return {"id": response_id, "text": full_text}


def non_stream_response(input_text: str, tools: list, previous_id: str = None) -> dict:
    """Make a non-streaming request."""
    payload = {
        "model": MODEL,
        "input": input_text,
        "tools": tools,
        "stream": False,
    }
    if previous_id:
        payload["previous_response_id"] = previous_id

    resp = requests.post(
        f"{BASE_URL}/v1/responses",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    return resp.json()


def print_streaming_response(input_text: str, tools: list, previous_id: str = None) -> str:
    """Print streaming response and return response ID."""
    response_id = None
    current_phase = None

    print("\n\033[90m", end="")  # Gray for reasoning

    for event in stream_response(input_text, tools, previous_id):
        event_type = event.get('type', '')

        # Track response ID
        if event_type == 'response.created':
            response_id = event.get('response', {}).get('id')

        # Reasoning (gray)
        elif event_type == 'response.reasoning_text.delta':
            if current_phase != 'reasoning':
                current_phase = 'reasoning'
                print("\033[90m[thinking] ", end="")
            print(event.get('delta', ''), end='', flush=True)

        # Code being generated (yellow)
        elif event_type == 'response.code_interpreter_call_code.delta':
            if current_phase != 'code':
                current_phase = 'code'
                print("\033[0m\n\033[93m[code] ", end="")
            print(event.get('delta', ''), end='', flush=True)

        # Code execution
        elif event_type == 'response.code_interpreter_call.interpreting':
            print("\033[0m\n\033[92m>>> Executing code...\033[0m", flush=True)
            current_phase = 'executing'

        elif event_type == 'response.code_interpreter_call.completed':
            print("\033[92m>>> Done\033[0m", flush=True)

        # Function call (cyan)
        elif event_type == 'response.function_call_arguments.delta':
            if current_phase != 'function':
                current_phase = 'function'
                print("\033[0m\n\033[96m[function call] ", end="")
            print(event.get('delta', ''), end='', flush=True)

        # Final text (white)
        elif event_type == 'response.output_text.delta':
            if current_phase != 'text':
                current_phase = 'text'
                print("\033[0m\n\n", end="")
            print(event.get('delta', ''), end='', flush=True)

        # Output item added
        elif event_type == 'response.output_item.added':
            item = event.get('item', {})
            item_type = item.get('type')
            if item_type == 'function_call':
                print(f"\033[0m\n\033[96m[Tool Call: {item.get('name')}]\033[0m", flush=True)

    print("\033[0m\n")  # Reset color
    return response_id


def print_non_streaming_response(response: dict) -> str:
    """Print non-streaming response and return response ID."""
    response_id = response.get('id')

    for item in response.get('output', []):
        item_type = item.get('type')

        if item_type == 'reasoning':
            content = item.get('content', [])
            if content:
                text = content[0].get('text', '')
                print(f"\033[90m[thinking] {text[:200]}{'...' if len(text) > 200 else ''}\033[0m")

        elif item_type == 'code_interpreter_call':
            code = item.get('code', '')
            outputs = item.get('outputs', [])
            print(f"\033[93m[code executed]\n{code}\033[0m")
            if outputs:
                print(f"\033[92m[output] {outputs}\033[0m")

        elif item_type == 'function_call':
            print(f"\033[96m[Tool Call: {item.get('name')}]")
            print(f"  Arguments: {item.get('arguments')}\033[0m")
            print("\n  (You need to execute this tool and provide the result)")

        elif item_type == 'message':
            for c in item.get('content', []):
                if c.get('type') == 'output_text':
                    print(f"\n{c.get('text', '')}")

    # Show token usage
    usage = response.get('usage', {})
    details = usage.get('output_tokens_details', {})
    tool_tokens = details.get('tool_output_tokens', 0)
    if tool_tokens > 0:
        print(f"\n\033[90m[tool_output_tokens: {tool_tokens}]\033[0m")

    print()
    return response_id


def main():
    parser = argparse.ArgumentParser(description="Chat with gpt-oss-20b using tools")
    parser.add_argument('--no-stream', action='store_true', help="Disable streaming")
    parser.add_argument('--no-code', action='store_true', help="Disable code_interpreter")
    parser.add_argument('--no-search', action='store_true', help="Disable web_search")
    args = parser.parse_args()

    # Check server
    try:
        resp = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error: Cannot connect to vLLM server at {BASE_URL}")
        print(f"Make sure server is running: ./serve_gpt_oss.sh")
        sys.exit(1)

    # Build tools list
    tools = []
    if not args.no_code:
        tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
    if not args.no_search:
        tools.append({"type": "web_search_preview"})

    print("=" * 60)
    print("  Chat with gpt-oss-20b (vLLM /v1/responses API)")
    print("=" * 60)
    print(f"  Tools: {[t['type'] for t in tools] if tools else 'None'}")
    print(f"  Streaming: {not args.no_stream}")
    print()
    print("  Commands:")
    print("    /quit or /exit  - Exit the chat")
    print("    /clear          - Clear conversation history")
    print("    /tools          - Toggle tools on/off")
    print("    /code <python>  - Execute Python directly")
    print("=" * 60)
    print()

    previous_id = None
    use_tools = True

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ['/quit', '/exit', '/q']:
            print("Goodbye!")
            break

        if user_input.lower() == '/clear':
            previous_id = None
            print("Conversation cleared.\n")
            continue

        if user_input.lower() == '/tools':
            use_tools = not use_tools
            print(f"Tools {'enabled' if use_tools else 'disabled'}.\n")
            continue

        if user_input.lower().startswith('/code '):
            code = user_input[6:]
            user_input = f"Execute this Python code and show me the exact output:\n```python\n{code}\n```"

        # Make request
        active_tools = tools if use_tools else []

        print("\033[95mAssistant:\033[0m", end="")

        try:
            if args.no_stream:
                response = non_stream_response(user_input, active_tools, previous_id)
                previous_id = print_non_streaming_response(response)
            else:
                previous_id = print_streaming_response(user_input, active_tools, previous_id)
        except Exception as e:
            print(f"\n\033[91mError: {e}\033[0m\n")
            continue


if __name__ == "__main__":
    main()
