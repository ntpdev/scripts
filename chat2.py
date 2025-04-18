#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import platform

# import sympy # used by eval
import re
from dataclasses import asdict, dataclass

import requests
import yaml
from dataclasses_json import dataclass_json
from firecrawl import FirecrawlApp

# from datetime import datetime, date, time
from openai import OpenAI
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint
from ftutils import ftutils_functions  # retrieve_headlines, retrieve_article, retrieve_stock_quotes, get_function_map

from chatutils import (
    CodeBlock,
    execute_script,
    extract_code_block,
    input_multi_line,
    make_fullpath,
    save_content,
    translate_latex,
)

# pip install dataclasses-json
# OpenAI Python library: https://github.com/openai/openai-python
# togetherAI models https://docs.together.ai/docs/chat-models
def get_model_info():
    """Parse the model info CSV into a dictionary using dict comprehension."""
    data = """
# key name provider
gptm gpt-4.1-mini openai
gpt4 gpt-4.1 openai
gpt45 gpt-4.5-preview openai
o4m o4-mini openai
groq llama-3.3-70b-versatile groq
#llm4 meta-llama/llama-4-scout-17b-16e-instruct groq
groq-r1 deepseek-r1-distill-llama-70b groq
# qwen qwen-2.5-coder-32b groq
qwq qwen-qwq-32b groq
llama meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 togetherai
llama-big meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo togetherai
ds deepseek-ai/DeepSeek-V3 togetherai
qwen Qwen/Qwen2.5-72B-Instruct-Turbo togetherai
samba DeepSeek-V3-0324 sambanova
samba-r1 DeepSeek-R1 sambanova
llama4 Llama-4-Maverick-17B-128E-Instruct sambanova
llama3 Meta-Llama-3.3-70B-Instruct sambanova
ollama llama3.1:8b-instruct-q5_K_M ollama
"""

    def filter_valid_lines(s):
        for line in s.splitlines():
            x = line.strip()
            if x and not x.startswith('#'):
                yield x

    def map_to_values(lines):
        for line in lines:
            parts = line.split(maxsplit=2)
            if len(parts) == 3:
                yield parts

    lines = filter_valid_lines(data)
    values = map_to_values(lines)
    return {key: {'name': name, 'provider': provider} for key, name, provider in values}


FNAME = "chat-log.json"
console = Console()
model_info = get_model_info()
code1 = None
role_to_color = {"system": "red", "developer": "red", "user": "green", "assistant": "cyan", "tool": "yellow"}
FNCALL_SYSMSG = """

You are Marvin, an AI chatbot trained by OpenAI. You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: 
<tools>{"type": "function", "function": {"name": "eval", "description": "evaluates a mathematical expression and returns the result example 5 * 4 + 3 .\n\n    Args:\n    code (str): a mathematical expression.\n\n    Returns:\n    str: a number.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}</tools>
You call a tool by outputting json within <tool_call> tags. The result will be provided to you within <tool_return> tags.
"""

tool_ex = """
<tool_call>
{'arguments': {'code': '(datetime.now() - datetime(1969, 7, 20)).days // 365'}, 'name': 'eval'}
</tool_call>
"""


q = """```python
from math import sqrt
x = 3
y = 4
print(sqrt(x ** 2 + y ** 2))
```
"""


@dataclass_json
@dataclass
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("Role cannot be empty")
        # if not self.content:
        #     raise ValueError("Content cannot be empty")


@dataclass_json
@dataclass
# add validation to check that role and content are not empty
class ChatToolMessageResponse(ChatMessage):
    name: str
    tool_call_id: str

    def __init__(self, name: str, tool_call_id: str, content: str):
        super().__init__("tool", content)
        self.name = name
        self.tool_call_id = tool_call_id


@dataclass_json
@dataclass
class ChatToolMessageCall(ChatMessage):
    tool_calls: list

    def __init__(self, chat_completion):
        """takes openAI ChatCompletionMessageToolCall and saves tool_calls"""
        super().__init__("assistant", None)
        self.tool_calls = chat_completion.to_dict()["tool_calls"]


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    prompt_cost: float
    completion_cost: float

    def __init__(self, prompt_c: float, completion_c: float):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.prompt_cost = prompt_c
        self.completion_cost = completion_c

    def cost(self) -> float:
        return (self.prompt_tokens * self.prompt_cost + self.completion_tokens * self.completion_cost) / 1e6

    def update(self, prompt_c: int, completion_c: int):
        self.prompt_tokens += prompt_c
        self.completion_tokens += completion_c


class LLM:

    def __init__(self, llm_name: str, use_tool: bool = False):
        # o1 model does not support tool use or temperature
        self.llm_name = llm_name
        self.model = model_info.get(llm_name, "local") # the default wont work
        self.client = self.create_client(llm_name)
        self.use_tool = use_tool

    def __str__(self):
        return f"{self.model} {self.llm_name} tool use = {self.use_tool}"

    def create_client(self, llm_name):
        provider = model_info[llm_name]["provider"]
        if provider == "openai":
            return OpenAI()
        if provider == "togetherai":
            return OpenAI(api_key=os.environ["TOGETHERAI_API_KEY"], base_url="https://api.together.xyz/v1")
        if provider == "groq":
            return OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
        if provider == "sambanova":
            return OpenAI(api_key=os.environ["SAMBANOVA_API_KEY"], base_url="https://api.sambanova.ai/v1")
        if provider == "ollama":
            return OpenAI(api_key="dummy", base_url="http://localhost:11434/v1")
        # lmstudio port
        return OpenAI(api_key="dummy", base_url="http://localhost:1234/v1")

    def chat(self, messages):
        nm = self.model["name"]
        s = nm.lower()
        is_reasoning = s == "o3-mini" or "r1" in s or "qwq" in s
        supportsTemp = nm != "o3-mini"

        args = {
            "model": nm,
            "messages": [asdict(m) for m in messages],
            "max_completion_tokens": 16000 if is_reasoning else 4096
        }

        if self.use_tool:
            args["tools"] = [v["defn"] for v in ftutils_functions().values()]
            if supportsTemp:
                args["temperature"] = 0.2
        elif supportsTemp:
            args["temperature"] = 1

        return self.client.chat.completions.create(**args)

    def toggle_tool_use(self) -> bool:
        self.use_tool = not self.use_tool
        return self.use_tool

tokens = Usage(0.15, 0.60)


def is_toolcall(s: str) -> str:
    console.print(s, style="yellow")
    start = s.find("<tool_call>")
    if start >= 0:
        end = s.find("</tool_call>")
        contents = s[start + 11 : end].strip()
        contents = contents.replace("'", '"')
        print(contents)
        #        data = json.loads(contents)
        #        print(data['code'])
        x = "<tool_response>" + eval("datetime.now()").__repr__() + "</tool_response>"
        print(x)
        return x
    return None


def prt(msg: ChatMessage):
    c = role_to_color[msg.role]
    console.print(f"{msg.role}:\n", style=c)
    md = Markdown(translate_latex(msg.content))
    console.print(md, style=c, width=80)


def save(xs, filename):
    with open(filename, "w") as f:
        f.write(ChatMessage.schema().dumps(xs, many=True))


def load_msg(s: str) -> ChatMessage:
    # this function used to role = "assistant" if len(xs) > 2 else "user"
    fname = make_fullpath(s)
    role = "user"

    try:
        with open(fname, encoding="utf-8") as f:
            return ChatMessage(role, f.read())
    except FileNotFoundError:
        console.print(f"{fname} FileNotFoundError", style="red")
    #       raise FileNotFoundError(f"Chat message file not found: {filename}")
    return None


def load_textfile(s: str) -> str:
    """loads a text file. if it is a code file wraps in markdown code block."""
    fname = make_fullpath(s)
    console.print(f"loading file {fname}")
    try:
        with open(fname, encoding="utf-8") as f:
            content = f.read()
            console.print(f"loaded file {fname} length {len(content)}", style="yellow")
            if fname.suffix in (".py", ".htm", ".html", ".java"):
                language = fname.suffix[1:] if fname.suffix != ".htm" else "html"
                return f"\n## {fname.name}\n\n```{language}\n{content}\n```\n"
            return content
    except FileNotFoundError:
        console.print(f"{fname} FileNotFoundError", style="red")
    return None


def load_template(s: str) -> ChatMessage:
    xs = s.split(maxsplit=1)
    fname = make_fullpath(xs[0])
    rprint(xs)

    try:
        with open(fname, encoding="utf-8") as f:
            templ = f.read()
            if len(xs) > 1:
                templ = templ.replace("{input}", xs[1])

            return ChatMessage("user", templ)
    except FileNotFoundError:
        console.print(f"{fname} FileNotFoundError", style="red")
    s = """
**question:**

{input}

**instructions:**
1. first contextualize and disambiguate the question. if necessary ask questions to further clarify.
2. write out your thoughts in a **thinking:** section
3. write a detailed answer with supporting evidence and facts in an **answer:** section
"""
    return ChatMessage("user", s.replace("{input}", xs[1]))


#       raise FileNotFoundError(f"Chat message file not found: {filename}")
# return None


def load_log(s: str) -> list[ChatMessage]:
    xs = s.split()
    fname = make_fullpath(xs[1])

    try:
        with open(fname) as f:
            data = json.load(f)
            all_msgs = ChatMessage.schema().load(data, many=True)
            console.print(f"loaded from log {len(xs)} messages {len(all_msgs)}", style="red")
            #            save_content(all_msgs[-1])
            xs = all_msgs if len(all_msgs) < 20 else all_msgs[:3] + all_msgs[-20:]
            prt_summary(xs)
            return xs
    except FileNotFoundError:
        console.print(f"{fname} FileNotFoundError", style="red")
    #       raise FileNotFoundError(f"Chat message file not found: {filename}")
    except json.JSONDecodeError:
        console.print(f"{fname} JSONDecodeError", style="red")
    #        raise JSONDecodeError(f"Error parsing JSON data in {filename}")
    return None


def make_clean_filename(text: str) -> str:
    words = re.sub(r"[\\\.\/[\]<>'\":*?|]", " ", text.lower()).split()
    return "_".join(words[:5])


def load_http(url: str) -> ChatMessage:
    try:
        crawler = FirecrawlApp(api_key=os.environ["FC_API_KEY"])
        result = crawler.scrape_url(url, params={"formats": ["markdown"]})
        #        pprint(result)
        if result["metadata"]["statusCode"] == 200:
            text = result["markdown"]
            title = result["metadata"]["title"]
            fname = make_clean_filename(title) + ".md"
            console.print(f"saving {url}\n to {fname}", style="red")
            markdown = Markdown(text, style="yellow", code_theme="monokai")
            console.print(markdown, width=80)
            with open(make_fullpath(fname), "w", encoding="utf-8") as f:
                f.write(f"[*source* {result['metadata']['title']}]({url})\n\n")
                f.write(text)
            return ChatMessage("user", text)

    except Exception as e:  # requests.exceptions.RequestException as e:
        print(f"Error: An error occurred while fetching the webpage: {e}")

    return None


def prt_summary(msgs: list[ChatMessage]):
    cs = [(len(msg.content)) for msg in msgs]
    ws = [(len(msg.content.split())) for msg in msgs]

    console.print(f"loaded from log {len(msgs)} words {sum(ws)} chars {sum(cs)}", style="red")
    for i, m in enumerate(msgs):
        c = m.content.replace("\n", "\\n")  # msgs:
        s = f"{i:2} {m.role:<10} {c if len(c) < 70 else c[:70] + ' ...'}"
        console.print(s, style=role_to_color[m.role])


def tool_response(s: str) -> str:
    xs = s[6:].strip()
    return '<tool_response>\n{"name": "eval", "content": "xxx"}\n</tool_response>\n'.replace("xxx", xs)


def load(filename: str) -> list[ChatMessage]:
    with open(filename) as f:
        return ChatMessage.schema().parse_raw(f.read())


def process_tool_call(tool_call):
    fnname = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    console.print(f"tool call {fnname} {args}", style="yellow")

    if fnname in ftutils_functions():
        try:
            r = ftutils_functions()[fnname]["fn"](**args)
            # tool return value is either a string or a pydantic data type
            if isinstance(r, str):
                markdown = Markdown(r, style="yellow", code_theme="monokai")
                console.print(markdown, width=80)
                return ChatToolMessageResponse(fnname, tool_call.id, r)
            console.print(f"result = {r}", style="yellow")
        except Exception as e:
            r = f"ERROR: {e}"
            console.print(r, style="red")
            return ChatToolMessageResponse(fnname, tool_call.id, r)

        return ChatToolMessageResponse(fnname, tool_call.id, r.model_dump_json())

    err_msg = f"ERROR: unknown funtion name " + fnname
    console.print(err_msg, style="red")
    return ChatToolMessageResponse(fnname, tool_call.id, err_msg)


def check_and_process_tool_call(client, messages, response):
    """check for a tool call and process. If there is no tool call then the original response is returned"""
    # https://platform.openai.com/docs/guides/function-calling
    choice = response.choices[0]
    n = 5
    while choice.finish_reason == "tool_calls" and n > 0:
        n -= 1
        # append choice.message to message history
        messages.append(ChatToolMessageCall(choice.message))
        for tc in choice.message.tool_calls:
            tcm = ChatToolMessageResponse(tc.function.name, tc.id, tc.function.arguments)
            prt(tcm)
            tool_response = process_tool_call(tc)
            messages.append(tool_response)
        # reply to llm with tool responses
        response = client.chat(messages)
        choice = response.choices[0]

    if n == 0:
        console.print("tool call limit exceeded", style="red")
    return response


def check_and_process_code_block(client, messages, response):
    """check for a code block, execute it and pass output back to LLM. This can happen several times if there are errors. If there is no code block then the original response is returned"""
    code = extract_code_block_from_response(response)
    n = 0
    while code and len(code.language) > 0 and n < 5:
        # store original message from llm
        m = response.choices[0].message
        msg = ChatMessage(m.role, m.content)
        messages.append(msg)
        prt(msg)
        output = execute_script(code)
        n += 1
        code = None
        if output:
            msg2 = ChatMessage("user", "## output from running script\n" + output + "\n")
            messages.append(msg2)
            prt(msg2)
            response = client.chat(messages)
            code = extract_code_block_from_response(response)
            ru = response.usage
            tokens.update(ru.prompt_tokens, ru.completion_tokens)
            pprint(tokens)

    return response


def extract_code_block_from_response(response) -> CodeBlock:
    return extract_code_block(response.choices[0].message.content, "```")


def process_commands(client: LLM, cmd: str, inp: str, messages: list[ChatMessage]) -> bool:
    global code1
    next_action = False
    if cmd == "load":
        msg = load_msg(inp)
        if msg:
            messages.append(msg)
            prt(msg)
            next_action = msg.role == "user"
    elif cmd == "tmpl":
        msg = load_template(inp)
        if msg:
            messages.append(msg)
            prt(msg)
            next_action = True
    if cmd == "web":
        msg = load_http(inp)
        if msg:
            messages.append(msg)
            next_action = True
    if cmd == "code":
        code1 = load_textfile(inp)
    elif cmd == "resp":
        msg = ChatMessage("user", tool_response(inp))
    elif cmd == "reset":
        messages.clear()
        messages.append(ChatMessage("system", system_message()))
    elif cmd == "drop":
        # remove last response for LLM and user msg that triggered
        if len(messages) > 2:
            messages.pop()
            messages.pop()
    elif cmd == "log":
        messages.clear()
        xs = load_log(inp)
        for x in xs:
            messages.append(x)
        next_action = messages[-1].role == "user"
    elif cmd == "save":
        save_content(messages[-1].content)
    elif cmd == "tool":
        state = client.toggle_tool_use()
        console.print(f"tool use changed to {state}", style="yellow")
    return next_action


def system_message():
    tm = datetime.datetime.now().isoformat()
    scripting_lang, plat = ("bash", "Ubuntu") if platform.system() == "Linux" else ("powershell", "Windows 11")
    return f"you are Marvin a super intelligent AI assistant. You provide accurate information. If you are unsure or don't have the correct information say so. The current datetime is {tm}."


def chat(llm_name, use_tool):
    # useTool pass llm_name.startswith('gpt')
    global code1
    client = LLM(llm_name, use_tool)
    #    systemMessage = ChatMessage('system', FNCALL_SYSMSG)
    system_msg = ChatMessage("system", system_message())
    rprint(system_msg)
    messages = [] if llm_name.startswith("o") else [system_msg]
    print(f"chat with {client}. Enter x to exit.")
    inp = ""
    while inp != "x":
        inp = input_multi_line()
        if len(inp) > 3:
            if inp.startswith("%"):
                cmds = inp.split(maxsplit=1)
                if not process_commands(client, cmds[0][1:], cmds[1] if len(cmds) > 1 else None, messages):
                    continue
            else:
                if code1:
                    msg = ChatMessage(f"user", f"{inp}\n{code1}")
                    code1 = None
                else:
                    msg = ChatMessage("user", inp)
                messages.append(msg)
                prt(msg)
            response = client.chat(messages)
            response = check_and_process_tool_call(client, messages, response)
            response = check_and_process_code_block(client, messages, response)
            # store original message from gpt
            m = response.choices[0].message
            if m.content:
                msg = ChatMessage(m.role, m.content)
                messages.append(msg)
                prt(msg)

            ru = response.usage
            tokens.update(ru.prompt_tokens, ru.completion_tokens)
            pprint(ru)
            pprint(tokens)
            print(f"prompt tokens: {ru.prompt_tokens}, completion tokens: {ru.completion_tokens}, total tokens: {ru.total_tokens} cost: {tokens.cost():.4f}")

    if len(messages) > 2:
        save(messages, make_fullpath(FNAME))
        yaml.dump(messages, open(make_fullpath("chat-log.yaml"), "w"))


def chat_ollama():
    url = "http://localhost:11434/api/chat"
    messages = []
    messages.append(
        {
            "role": "system",
            "content": "You are Marvin. You use logic and reasoning when answering questions. Answer accurately, concisely.",
        }
    )
    #    messages.append({'role': 'user', 'content':'role: physics professor. question: what is the Hall effect? style: undergraduate lecture'})
    messages.append({"role": "user", "content": load_msg("%load Koopa.txt")})
    data = {"model": "llama3", "messages": messages, "stream": True}
    print("call chat")
    response = requests.post(url, json=data)
    output = ""
    message = {}

    for line in response.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            # the response streams one token at a time, print that as we receive it
        #            print(content, end="", flush=True)

        if body.get("done", False):
            message["content"] = output
            return message
    return message


def chat_ollama2():
    """call ollama using generate endpoint"""
    url = "http://localhost:11434/api/generate"
    data = {"model": "llama3:text", "prompt": load_msg("%load Koopa.txt"), "stream": True}
    print("call generate\n" + data["prompt"])
    response = requests.post(url, json=data)
    output = ""

    for line in response.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            content = body["response"]
            output += content
            # the response streams one token at a time, print that as we receive it
            print(content, end="", flush=True)

        if body.get("done", False):
            print("\n\n")
            return output
    return output


def x():
    print(system_message())
    s = """
here is some python code
```
dir $HOME/documents/*.txt
echo "hello world"
```
text after
"""
    c = extract_code_block(s, "```")
    console.print(yaml.dump(c), style="green")
    # execute_script(CodeBlock('powershell', s.split('\n')))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with LLMs")
    parser.add_argument(
        "llm",
        choices=list(model_info.keys()),
        type=str,
        help="LLM to use [local|ollama|gptm|gpt4|gpt45|o4m|llama|llama-big|qwen|ds|groq|groq-r1]",
    )
    parser.add_argument("tool_use", type=str, nargs="?", default="", help="add tool to enable tool calls")

    args = parser.parse_args()
    chat(args.llm, args.tool_use == "tool")
    # x()
    # s = chat_ollama()
    # print(s)
    # xs = load(FNAME)
