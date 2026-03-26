import re
import json

from agent.llm import get_response_from_llm, is_slm, SLM_MAX_TOKENS, MAX_TOKENS

# SLMs produce short outputs, so a much lower length triggers the truncation retry
_LLM_RETRY_THRESHOLD = 2000
_SLM_RETRY_THRESHOLD = 1024

# SLMs are more fragile with many sequential tool calls
_LLM_MAX_TOOL_CALLS = 40
_SLM_MAX_TOOL_CALLS = 10


def get_tooluse_prompt(tool_infos=[]):
    """
    Get the prompt for using the available tools.
    """
    if not tool_infos or len(tool_infos) == 0:
        return ""
    tools_available = [str(tool_info) for tool_info in tool_infos]
    tools_available = '\n\n'.join(tools_available) if tools_available else 'None'
    tooluse_prompt = """Here are the available tools:
```
{tools_available}
```

Use only one tool (if needed) in this format:
<json>
{{
    "tool_name": ...,
    "tool_input": ...
}}
</json>

ONLY USE ONE TOOL PER RESPONSE, AND STRICTLY FOLLOW THE FORMAT OF TOOL_NAME AND TOOL_INPUT ABOVE.
DO NOT HALLUCINATE OR MAKE UP ANYTHING.
""".format(tools_available=tools_available)
    return tooluse_prompt.strip()


def should_retry_tool_use(response, tool_uses=None, slm=False):
    """
    Check if the response attempts to use a tool but ran out of output context.
    Uses a lower length threshold for SLMs since their max output is much shorter.
    """
    if tool_uses is not None and len(tool_uses) > 0:
        return False

    json_pos       = response.find("<json>")
    tool_name_pos  = response.find("tool_name")
    tool_input_pos = response.find("tool_input")

    threshold = _SLM_RETRY_THRESHOLD if slm else _LLM_RETRY_THRESHOLD

    if (
        json_pos != -1
        and tool_name_pos != -1
        and tool_input_pos != -1
        and json_pos < tool_name_pos < tool_input_pos
        and len(response) >= threshold
    ):
        return True

    return False


def check_for_tool_uses(response):
    """
    Checks if the response contains one or more tool calls in json code blocks.
    Returns a list of tool use dictionaries.
    """
    pattern = r'<json>\s*(\{.*?\})\s*</json>'
    matches = re.findall(pattern, response, re.DOTALL)
    tool_uses = []

    for match in matches:
        try:
            tool_use = json.loads(match)
            if 'tool_name' not in tool_use or 'tool_input' not in tool_use:
                continue
            tool_uses.append(tool_use)
        except json.JSONDecodeError:
            continue

    return tool_uses if tool_uses else None


def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


def chat_with_agent(
    msg,
    model="claude-4-sonnet-genai",
    msg_history=None,
    logging=print,
    tools_available=[],       # Empty list means no tools, 'all' means all tools
    multiple_tool_calls=False,
    max_tool_calls=None,      # None → auto: 10 for SLMs, 40 for large models
    max_tokens=None,          # None → auto: SLM_MAX_TOKENS for SLMs, MAX_TOKENS for large
):
    from agent.tools import load_tools

    slm = is_slm(model)
    effective_max_tool_calls = max_tool_calls if max_tool_calls is not None \
        else (_SLM_MAX_TOOL_CALLS if slm else _LLM_MAX_TOOL_CALLS)
    effective_max_tokens = max_tokens if max_tokens is not None \
        else (SLM_MAX_TOKENS if slm else MAX_TOKENS)

    get_response_fn = get_response_from_llm
    if msg_history is None:
        msg_history = []
    new_msg_history = msg_history

    try:
        all_tools = load_tools(logging=logging, names=tools_available)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        system_msg = f"{get_tooluse_prompt([tool['info'] for tool in all_tools])}\n\n"
        num_tool_calls = 0

        logging(f"Input: {repr(msg)}")
        response, new_msg_history, info = get_response_fn(
            msg=system_msg + msg,
            model=model,
            msg_history=new_msg_history,
            max_tokens=effective_max_tokens,
        )
        logging(f"Output: {repr(response)}")

        tool_uses = check_for_tool_uses(response)
        retry_tool_use = should_retry_tool_use(response, tool_uses, slm=slm)
        while tool_uses or retry_tool_use:
            if effective_max_tool_calls > 0 and num_tool_calls >= effective_max_tool_calls:
                logging("Error: Maximum number of tool calls reached.")
                break

            tool_msgs = []

            if tool_uses:
                tool_uses = tool_uses if multiple_tool_calls else tool_uses[:1]
                for tool_use in tool_uses:
                    tool_name  = tool_use['tool_name']
                    tool_input = tool_use['tool_input']
                    tool_output = process_tool_call(tools_dict, tool_name, tool_input)
                    num_tool_calls += 1
                    tool_msg = f'''<json>
    {{
        "tool_name": "{tool_name}",
        "tool_input": {tool_input},
        "tool_output": "{tool_output}"
    }}
    </json>'''.strip()
                    logging(f"Tool output: {repr(tool_msg)}")
                    tool_msgs.append(tool_msg)

            if retry_tool_use:
                logging("Error: Output context exceeded. Please try again.")
                tool_msgs.append("Error: Output context exceeded. Please try again.")

            response, new_msg_history, info = get_response_fn(
                msg=system_msg + '\n\n'.join(tool_msgs),
                model=model,
                msg_history=new_msg_history,
                max_tokens=effective_max_tokens,
            )
            logging(f"Output: {repr(response)}")

            tool_uses = check_for_tool_uses(response)
            retry_tool_use = should_retry_tool_use(response, tool_uses, slm=slm)

    except Exception as e:
        logging(f"Error: {str(e)}")
        raise e

    return new_msg_history


if __name__ == "__main__":
    msg = """hello"""
    new_msg_history = chat_with_agent(msg)
