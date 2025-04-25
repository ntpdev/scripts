#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import platform

# import sympy # used by eval
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Type

import requests
import yaml
from dataclasses_json import dataclass_json
from firecrawl import FirecrawlApp

# from datetime import datetime, date, time
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint
from rich import inspect
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
@dataclass(kw_only=True)
class ContentItem:
    type: Literal["text"] = "text"
    text: str

@dataclass_json  
@dataclass
class FunctionCall:
    name: str
    arguments: str

@dataclass_json  
@dataclass
class ToolCall:
    id: str
    type: Literal["function"]
    function: FunctionCall


class Message:
    """Base class with serialization logic"""
    _role_registry: dict[str, Type['Message']] = {}
    
    def __init_subclass__(cls, **kwargs):
        """Register subclasses by their role"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'role'):
            role = getattr(cls, 'role')
            if isinstance(role, str):
                cls._role_registry[role] = cls
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Message':
        """Create appropriate message subclass from dict"""
        role = data.get('role')
        msg_cls = cls._role_registry.get(role)
        if not msg_cls:
            raise ValueError(f"Unknown message role: {role}")
        # Deserialize nested content dicts into ContentItem instances
        if 'content' in data and isinstance(data['content'], list):
            data['content'] = [ContentItem(**ci) for ci in data['content']]
        # Deserialize nested tool_calls dicts into ToolCall instances (with FunctionCall)
        if 'tool_calls' in data and isinstance(data['tool_calls'], list):
            converted_calls: list[ToolCall] = []
            for tc in data['tool_calls']:
                func_data = tc.get('function', {})
                func_obj = FunctionCall(**func_data)
                converted_calls.append(ToolCall(id=tc['id'], type=tc['type'], function=func_obj))
            data['tool_calls'] = converted_calls
        return msg_cls(**data)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert message to serializable dict"""
        return asdict(self)

    def get_content(self) -> str:
        if self.content is None:
            return ""
        return "".join(e.text for e in self.content) if isinstance(self.content, list) else self.content

@dataclass_json  
@dataclass(kw_only=True)
class SystemMessage(Message):
    role: Literal["system"] = "system"
    content: str | list[ContentItem]

@dataclass_json
@dataclass(kw_only=True)
class UserMessage(Message):
    role: Literal["user"] = "user"
    content: str | list[ContentItem]

@dataclass_json
@dataclass(kw_only=True)
class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"
    content: str | list[ContentItem] | None = None
    tool_calls: list[ToolCall] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass_json
@dataclass(kw_only=True)
class ToolMessage(Message):
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


@dataclass_json  
@dataclass 
class MessageHistory:
    """Stores conversation history with polymorphic message types"""
    
    messages: list[Message] = field(default_factory=list)
    
    def __iter__(self):
        return iter(self.messages)
    
    def __str__(self) -> str:
        """Return the number of messages in the history."""
        return f"MessageHistory: len={len(self.messages)}"

    def append(self, message: Message) -> None:
        self.messages.append(message)
    
    def clear(self) -> None:
        self.messages.clear()
    
    def save(self, fname: Path) -> None:
        json_str = self.to_json(indent=2)
        with fname.open('w', encoding="utf-8") as f:
            f.write(json_str)
    
    @classmethod
    def load(cls, fname: Path) -> 'MessageHistory':
        with fname.open(encoding="utf-8") as f:
            data = json.load(f)
            
            # Convert each message dict into the appropriate Message subclass
            messages = []
            for msg_data in data.get('messages', []):
                try:
                    messages.append(Message.from_dict(msg_data))
                except Exception as e:
                    console.print(f"Error loading message: {e}", style="red")
                    continue
            
            return cls(messages=messages)


def system_message(text: str) -> SystemMessage:
    return SystemMessage(content=text)

def user_message(text: str) -> UserMessage:
    return UserMessage(content=text)

def assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(content=text)

def assistant_message_tool(tool_calls: list[Any]) -> AssistantMessage:
    xs = [tool_call(e) for e in tool_calls] if isinstance(tool_calls[0], ChatCompletionMessageToolCall) else tool_calls
    return AssistantMessage(content=None, tool_calls=xs)

def tool_message(tool_call: ToolCall, result: str) -> ToolMessage:
    return ToolMessage(
        content=result, 
        tool_call_id=tool_call.id)

def tool_call(tc: ChatCompletionMessageToolCall) -> ToolCall:
    return ToolCall(
        id=tc.id,
        type=tc.type,
        function=FunctionCall(name=tc.function.name, arguments=tc.function.arguments))


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

    def chat(self, messages: MessageHistory) -> ChatCompletion:
        nm = self.model["name"]
        s = nm.lower()
        is_reasoning = s in ["o4-mini", "DeepSeek-R1", "qwen-qwq-32b"]
        supportsTemp = nm != "o4-mini"

        args = {
            "model": nm,
            "messages": [asdict(m) for m in messages],
            "max_completion_tokens": 16000 if is_reasoning else 4096
        }

        if self.use_tool:
            args["tools"] = [v["defn"] for v in ftutils_functions().values()]
            if supportsTemp:
                args["temperature"] = 0.6

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


def prt(msg: Message, model: str = None):
    c = role_to_color[msg.role]
    if model:
        console.print(f"{msg.role} ({model}):\n", style=c)
    else:
        console.print(f"{msg.role}:\n", style=c)
    md = Markdown(translate_latex(msg.get_content()))
    console.print(md, style=c, width=80)


def save(messages: list, filename: Path):
    """Save a list of messages to a JSON file, preserving all fields."""
    with filename.open("w", encoding="utf-8") as f:
        # Convert each message to dict using its own to_dict() method
        json.dump([msg.to_dict() for msg in messages], f)


def load_msg(s: str) -> UserMessage:
    # this function used to role = "assistant" if len(xs) > 2 else "user"
    fname = make_fullpath(s)
    #role = "user"

    try:
        with fname.open(encoding="utf-8") as f:
            return user_message(text=f.read())
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


def load_template(s: str) -> UserMessage:
    xs = s.split(maxsplit=1)
    fname = make_fullpath(xs[0])
    rprint(xs)

    try:
        with fname.open(encoding="utf-8") as f:
            templ = f.read()
            if len(xs) > 1:
                templ = templ.replace("{input}", xs[1])

            return user_message(text=templ)
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
    return user_message(text=s.replace("{input}", xs[1]))


#       raise FileNotFoundError(f"Chat message file not found: {filename}")
# return None


def load_log(s: str) -> MessageHistory:
    fname = make_fullpath(s)
    try:
        hist = MessageHistory.load(fname)
        console.print(f"loaded from log {fname} messages {hist}", style="yellow")
        # limit length to 10
        if len(hist.messages) > 10:
            hist.messages = hist.messages[:3] + hist.messages[-7:]
        prt_summary(hist)
        return hist
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


def load_http(url: str) -> UserMessage:
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
            return user_message(tex=text)

    except Exception as e:  # requests.exceptions.RequestException as e:
        print(f"Error: An error occurred while fetching the webpage: {e}")

    return None


def prt_summary(history: MessageHistory):
    xs = [msg.get_content() for msg in history]
    count_chars = sum((len(x)) for x in xs)
    count_words = sum((len(x.split())) for x in xs)

    console.print(f"loaded from log {history} words {count_words} chars {count_chars}", style="red")
    for i, m in enumerate(history):
        c = m.get_content().replace("\n", "\\n")  # msgs:
        s = f"{i:2} {m.role:<10} {c if len(c) < 70 else c[:70] + ' ...'}"
        console.print(s, style=role_to_color[m.role])


def tool_response(s: str) -> str:
    xs = s[6:].strip()
    return '<tool_response>\n{"name": "eval", "content": "xxx"}\n</tool_response>\n'.replace("xxx", xs)


# def load(filename: str) -> list[ChatMessage]:
#     with open(filename) as f:
#         return ChatMessage.schema().parse_raw(f.read())


def process_tool_call(tool_call: ChatCompletionMessageToolCall) -> ToolMessage:
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
                return tool_message(tool_call, r)
            console.print(f"result = {r}", style="yellow")
        except Exception as e:
            r = f"ERROR: {e}"
            console.print(r, style="red")
            return tool_message(tool_call, r)

        return tool_message(tool_call, r.model_dump_json())

    err_msg = f"ERROR: unknown funtion name " + fnname
    console.print(err_msg, style="red")
    return tool_message(tool_call, err_msg)


def check_and_process_tool_call(client: LLM, history: MessageHistory, response: ChatCompletion) -> ChatCompletion:
    """check for a tool call and process. If there is no tool call then the original response is returned"""
    # https://platform.openai.com/docs/guides/function-calling
    choice = response.choices[0]
    n = 9
    while choice.finish_reason == "tool_calls" and n:
        n -= 1
        # append choice.message to message history
        history.append(assistant_message_tool(choice.message.tool_calls))
        for tc in choice.message.tool_calls:
            tool_response = process_tool_call(tc)
            history.append(tool_response)
        # reply to llm with tool responses
        response = client.chat(history)
        choice = response.choices[0]

    if n == 0:
        console.print("consecutive tool call limit exceeded", style="red")
    return response


def check_and_process_code_block(client: LLM, history: MessageHistory, response: ChatCompletion) -> ChatCompletion:
    """check for a code block, execute it and pass output back to LLM. This can happen several times if there are errors. If there is no code block then the original response is returned"""
    code = extract_code_block_from_response(response)
    n = 3
    while code and len(code.language) > 0 and n:
        n -= 1
        # store original message from llm
        m = response.choices[0].message
        msg = assistant_message(text=m.content)
        history.append(msg)
        prt(msg)
        output = execute_script(code)
        code = None
        if output:
            msg2 = user_message(text="## output from running script\n" + output + "\n")
            history.append(msg2)
            prt(msg2)
            response = client.chat(history)
            code = extract_code_block_from_response(response)
            ru = response.usage
            tokens.update(ru.prompt_tokens, ru.completion_tokens)
            pprint(tokens)

    if n == 0:
        console.print("consecutive code block limit exceeded", style="red")
    return response


def extract_code_block_from_response(response: ChatCompletion) -> CodeBlock:
    return extract_code_block(response.choices[0].message.content, "```")


def process_commands(client: LLM, cmd: str, inp: str, history: MessageHistory) -> bool:
    global code1
    next_action = False
    if cmd == "load":
        msg = load_msg(inp)
        if msg:
            history.append(msg)
            prt(msg)
            next_action = msg.role == "user"
    elif cmd == "tmpl":
        msg = load_template(inp)
        if msg:
            history.append(msg)
            prt(msg)
            next_action = True
    if cmd == "web":
        msg = load_http(inp)
        if msg:
            history.append(msg)
            next_action = True
    if cmd == "code":
        code1 = load_textfile(inp)
    elif cmd == "resp":
        msg = user_message(text=tool_response(inp))
    elif cmd == "reset":
        history.clear()
        history.append(system_message(sys_msg()))
    elif cmd == "drop":
        # remove last response for LLM and user msg that triggered
        if len(history) > 2:
            history.pop()
            history.pop()
    elif cmd == "log":
        history.clear()
        xs = load_log(inp)
        for x in xs:
            history.append(x)
        next_action = history.messages[-1].role == "user"
    elif cmd == "save":
        save_content(history.messages[-1].get_content())
    elif cmd == "tool":
        state = client.toggle_tool_use()
        console.print(f"tool use changed to {state}", style="yellow")
    return next_action


def sys_msg():
    tm = datetime.datetime.now().isoformat()
    scripting_lang, plat = ("bash", "Ubuntu") if platform.system() == "Linux" else ("powershell", "Windows 11")
    return f"you are Marvin a super intelligent AI assistant. You provide accurate information. If you are unsure or don't have the correct information say so. The current datetime is {tm}."


def chat(llm_name, use_tool):
    # useTool pass llm_name.startswith('gpt')
    global code1
    client = LLM(llm_name, use_tool)
    #    systemMessage = ChatMessage('system', FNCALL_SYSMSG)
    history = MessageHistory()
    if not llm_name.startswith("o"):
        system_msg = system_message(sys_msg())
        rprint(system_msg)
        history.append(system_msg)
    print(f"chat with {client}. Enter x to exit.")
    inp = ""
    while inp != "x":
        inp = input_multi_line()
        if len(inp) > 3:
            if inp.startswith("%"):
                cmds = inp.split(maxsplit=1)
                if not process_commands(client, cmds[0][1:], cmds[1] if len(cmds) > 1 else None, history):
                    continue
            else:
                if code1:
                    msg = user_message(text=f"{inp}\n{code1}")
                    code1 = None
                else:
                    msg = user_message(text=inp)
                history.append(msg)
                prt(msg)
            response = client.chat(history)
            response = check_and_process_tool_call(client, history, response)
            response = check_and_process_code_block(client, history, response)
            reason = response.choices[0].finish_reason
            if reason != "stop":
                console.print(f"chat completion finish_reason {reason}", style="red")
            # store original message from gpt
            m = response.choices[0].message
            if m.content:
                msg = assistant_message(text=m.content)
                history.append(msg)
                prt(msg, response.model)

            ru = response.usage
            tokens.update(ru.prompt_tokens, ru.completion_tokens)
            # pprint(ru)
            # pprint(tokens)
            print(f"prompt tokens: {ru.prompt_tokens}, completion tokens: {ru.completion_tokens}, total tokens: {ru.total_tokens} cost: {tokens.cost():.4f}")

    if len(history.messages) > 2:
        history.save(make_fullpath(FNAME))
        yaml.dump(history, open(make_fullpath("chat-log.yaml"), "w"))


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

def test_message_history():
    history = MessageHistory()
    history.append(system_message(f"Hello {datetime.datetime.now().isoformat()}"))
    history.append(user_message("Hello"))
    history.append(assistant_message("Hi there!"))
    tc = ToolCall(id="123", type="function", function=FunctionCall(name="eval", arguments="2+3"))
    history.append(assistant_message_tool(tool_calls=[tc]))
    history.append(tool_message(tc, "5"))

    pprint(history.messages)
    # Save and load
    p = Path("chat.json")
    history.save(p)
    loaded_history = MessageHistory.load(p)
    pprint(str(loaded_history))
    for m in loaded_history:
        pprint(m)
    prt_summary(loaded_history)
    # Messages retain their types after loading
    assert isinstance(loaded_history.messages[0], SystemMessage)
    assert isinstance(loaded_history.messages[1], UserMessage)
    assert isinstance(loaded_history.messages[2], AssistantMessage)
    assert isinstance(loaded_history.messages[3], AssistantMessage)
    assert isinstance(loaded_history.messages[4], ToolMessage)

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
    # test_message_history()
    # exit(0)
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
