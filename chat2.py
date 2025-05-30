#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import platform
import textwrap

# import sympy # used by eval
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

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

from chatutils import (
    execute_script,
    extract_code_block,
    load_textfile,
    input_multi_line,
    make_fullpath,
    save_content,
    translate_latex,
    translate_thinking,
)
from ftutils import ftutils_functions  # retrieve_headlines, retrieve_article, retrieve_stock_quotes, get_function_map

LOG_FILE = make_fullpath("chat-log.json")
console = Console()
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

# OpenAI Python library: https://github.com/openai/openai-python
# togetherAI models https://docs.together.ai/docs/chat-models


@dataclass
class Provider:
    id: str
    env_var: str | None
    url: str

    def get_api_key(self) -> str:
        return os.getenv(self.env_var) if self.env_var else "dummy"


@dataclass
class ModelInfo:
    id: str
    name: str
    provider: Provider
    reasoning: bool = False


def load_model_info(file_path: str) -> dict[str, ModelInfo]:
    """Load the list of models and providers from yaml"""
    path = make_fullpath(file_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    providers = {provider_id: Provider(id=provider_id, **v) for provider_id, v in data["providers"].items()}

    for model in data["models"].values():
        model["provider"] = providers[model["provider"]]

    return {model_id: ModelInfo(id=model_id, **v) for model_id, v in data["models"].items()}


model_info = load_model_info("model-info.yaml")


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

    _role_registry: dict[str, type["Message"]] = {}

    def __init_subclass__(cls, **kwargs):
        """Register subclasses by their role"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "role"):
            role = getattr(cls, "role")
            if isinstance(role, str):
                cls._role_registry[role] = cls

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create appropriate message subclass from dict"""
        role = data.get("role")
        msg_cls = cls._role_registry.get(role)
        if not msg_cls:
            raise ValueError(f"Unknown message role: {role}")
        # Deserialize nested content dicts into ContentItem instances
        if "content" in data and isinstance(data["content"], list):
            data["content"] = [ContentItem(**ci) for ci in data["content"]]
        # Deserialize nested tool_calls dicts into ToolCall instances (with FunctionCall)
        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            converted_calls: list[ToolCall] = []
            for tc in data["tool_calls"]:
                func_data = tc.get("function", {})
                func_obj = FunctionCall(**func_data)
                converted_calls.append(ToolCall(id=tc["id"], type=tc["type"], function=func_obj))
            data["tool_calls"] = converted_calls
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
class DeveloperMessage(Message):
    role: Literal["developer"] = "developer"
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
        fname.write_text(json_str, encoding="utf-8")

    @classmethod
    def load(cls, fname: Path) -> "MessageHistory":
        data = json.loads(fname.read_text(encoding="utf-8"))

        # Convert each message dict into the appropriate Message subclass
        messages = []
        for msg_data in data.get("messages", []):
            try:
                messages.append(Message.from_dict(msg_data))
            except Exception as e:
                console.print(f"Error loading message: {e}", style="red")
                continue

        return cls(messages=messages)


def system_message(text: str) -> SystemMessage:
    return SystemMessage(content=text)

def developer_message(text: str) -> DeveloperMessage:
    return DeveloperMessage(content=text)

def user_message(text: str) -> UserMessage:
    return UserMessage(content=text)


def assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(content=text, tool_calls=None)


def assistant_tool_message(tool_calls: list[Any]) -> AssistantMessage:
    xs = [tool_call(e) for e in tool_calls] if isinstance(tool_calls[0], ChatCompletionMessageToolCall) else tool_calls
    return AssistantMessage(content=None, tool_calls=xs)


def tool_message(tool_call: ToolCall, result: str) -> ToolMessage:
    return ToolMessage(content=result, tool_call_id=tool_call.id)


def tool_call(tc: ChatCompletionMessageToolCall) -> ToolCall:
    return ToolCall(id=tc.id, type=tc.type, function=FunctionCall(name=tc.function.name, arguments=tc.function.arguments))


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
    llm_name: str
    model: ModelInfo
    client: OpenAI
    use_tool: bool

    def __init__(self, llm_name: str, use_tool: bool = False):
        self.llm_name = llm_name
        self.model = model_info[llm_name]
        self.use_tool = use_tool
        self.create_client()

    def __str__(self):
        return f"{self.model.name} on {self.model.provider.id} {self.llm_name} tool use = {self.use_tool}"

    def create_client(self) -> None:
        provider = self.model.provider
        self.client = OpenAI() if provider.id == "openai" else OpenAI(api_key=provider.get_api_key(), base_url=provider.url)
        # lmstudio port base_url="http://localhost:1234/v1"

    def chat(self, messages: MessageHistory) -> ChatCompletion:
        def clean_asdict(obj):
            # certain providers groq do not like tool_calls = None in assistant messages so remove None values
            data = asdict(obj)
            return {k: v for k, v in data.items() if v is not None}

        supports_temp = self.model.name != "o4-mini"

        args = {"model": self.model.name, "messages": [clean_asdict(m) for m in messages], "max_completion_tokens": 16000 if self.model.reasoning else 4096}

        if self.use_tool:
            args["tools"] = [v["defn"] for v in ftutils_functions().values()]
            if supports_temp:
                args["temperature"] = 0.6
        try:
            res = self.client.chat.completions.create(**args)
        except Exception as e:
            console.print(f"Error: {e}", style="red")
            breakpoint()
            raise
        return res

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
    text = translate_thinking(translate_latex(msg.get_content()))
    console.print(Markdown(text), style=c, width=80)


def save(messages: list, filename: Path):
    """Save a list of messages to a JSON file, preserving all fields."""
    with filename.open("w", encoding="utf-8") as f:
        # Convert each message to dict using its own to_dict() method
        json.dump([msg.to_dict() for msg in messages], f)


def load_msg(s: str) -> UserMessage:
    # this function used to role = "assistant" if len(xs) > 2 else "user"
    fname = make_fullpath(s)
    # role = "user"

    try:
        return user_message(text=fname.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        console.print(f"{e.__class__.__name__}: {e}", style="red")
    return None


def load_template(s: str) -> UserMessage:
    xs = s.split(maxsplit=1)
    fname = make_fullpath(xs[0])
    rprint(xs)

    try:
        templ = fname.read_text(encoding="utf-8")
        if len(xs) > 1:
            templ = templ.replace("{input}", xs[1])

        return user_message(text=templ)
    except FileNotFoundError as e:
        console.print(f"{e.__class__.__name__}: {e}", style="red")

    s = """\
**question:**

{input}

**instructions:**
1. first contextualize and disambiguate the question. if necessary ask questions to further clarify.
2. write out your thoughts in a **thinking:** section
3. write a detailed answer with supporting evidence and facts in an **answer:** section
"""
    return user_message(text=s.replace("{input}", xs[1]))


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
    except FileNotFoundError as e:
        console.print(f"{e.__class__.__name__}: {e}", style="red")
    except json.JSONDecodeError as e:
        console.print(f"{e.__class__.__name__}: {e} file=\"{fname}\"", style="red")
    return None


def make_clean_filename(text: str) -> str:
    words = re.sub(r"[\\/[\]<>'\":*?|,.]", " ", text.lower()).split()
    return "_".join(words[:5])


def process_markdown_hyperlinks(text: str) -> str:
    # regex to match [text](url)
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    links = []

    def replacer(match):
        label = match.group(1)
        url   = match.group(2)
        if label and url.startswith("http"):
            links.append((label, url))
            idx = len(links)
            # replace with italic label + reference [idx]
            return f'*{label}* [{idx}]'
        return match.group(0)

    # replace all occurrences
    result = pattern.sub(replacer, text)

    # if we found any links, append the list at the end
    if links:
        result += '\n\n## links\n'
        for i, (label, url) in enumerate(links, start=1):
            result += f'{i}. {label} {url}\n'

    return result


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
            processed_text = process_markdown_hyperlinks(text)
            markdown = Markdown(processed_text, style="yellow", code_theme="monokai")
            console.print(markdown, width=80)
            with open(make_fullpath(fname), "w", encoding="utf-8") as f:
                f.write(f"[*source* {result['metadata']['title']}]({url})\n\n")
                f.write(processed_text)
            return user_message(text=processed_text)

    except Exception as e:  # requests.exceptions.RequestException as e:
        console.print(f"{e.__class__.__name__}: {e} url=\"{url}\"", style="red")

    return None


def prt_summary(history: MessageHistory):
    xs = [msg.get_content() for msg in history]
    count_chars = sum((len(x)) for x in xs)
    count_words = sum((len(x.split())) for x in xs)

    console.print(f"loaded from log {history} words {count_words} chars {count_chars}", style="red")
    for i, m in enumerate(history):
        c = m.get_content().replace("\n", "\\n")  # msgs:
        short_c = textwrap.shorten(c, width=70)
        s = f"{i:2} {m.role:<10} {short_c}"
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
            r = f"ERROR: {e.__class__.__name__}: {e}"
            console.print(r, style="red")
            return tool_message(tool_call, r)

        return tool_message(tool_call, r.model_dump_json())

    err_msg = "ERROR: unknown funtion name " + fnname
    console.print(err_msg, style="red")
    return tool_message(tool_call, err_msg)


def check_and_process_tool_call(client: LLM, history: MessageHistory, response: ChatCompletion) -> ChatCompletion:
    """check for a tool call and process. If there is no tool call then the original response is returned"""
    # https://platform.openai.com/docs/guides/function-calling
    for choice in response.choices:
        n = 9
        while choice.finish_reason == "tool_calls" and n:
            n -= 1
            # append choice.message to message history
            history.append(assistant_tool_message(choice.message.tool_calls))
            for tc in choice.message.tool_calls:
                tool_response = process_tool_call(tc)
                history.append(tool_response)
            # reply to llm with tool responses
            response = client.chat(history)
            choice = response.choices[0]

        if n == 0:
            console.print("consecutive tool call limit exceeded", style="red")
    return response


def check_and_process_code_block(history: MessageHistory) -> bool:
    """check for a code block, execute it and add output as user msg"""
    if msg := next(e for e in reversed(history.messages) if e.role == "assistant"):
        code = extract_code_block(msg.get_content(), "```")
        if code and code.language:
            if output := execute_script(code):
                msg2 = user_message(text="## output from running script\n" + output + "\n")
                history.append(msg2)
                prt(msg2)
                return True
    return False


def process_commands(client: LLM, cmd: str, inp: str, history: MessageHistory) -> bool:
    """ret True if next action is to call model otherwise will wait for user input"""
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
        code1 = load_textfile(make_fullpath(inp))
    elif cmd == "resp":
        msg = user_message(text=tool_response(inp))
    elif cmd == "reset":
        history.clear()
        history.append(sys_msg(client.model.name))
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
    elif cmd == "exec":
        next_action = check_and_process_code_block(history)

    return next_action


def sys_msg(model_name: str) -> Message:
    tm = datetime.datetime.now().isoformat()
    scripting_lang, plat = ("bash", "Ubuntu") if platform.system() == "Linux" else ("powershell", "Windows 11")
    m = f"you are Marvin a super intelligent AI assistant. You provide accurate information. If you are unsure or don't have the correct information say so. The current datetime is {tm}."
    return developer_message(m) if model_name.startswith("o") else system_message(m)


def chat(llm_name, use_tool):
    # useTool pass llm_name.startswith('gpt')
    global code1
    client = LLM(llm_name, use_tool)
    history = MessageHistory()
    history.append(sys_msg(client.model.name))
    pprint(history)
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
            reason = response.choices[0].finish_reason
            if reason != "stop":
                console.print(f"chat completion finish_reason {reason}", style="red")
            # store original message from gpt
            txt = "".join(c.message.content for c in response.choices if c.message.content)
            if txt:
                msg = assistant_message(text=txt)
                history.append(msg)
                prt(msg, response.model)

            ru = response.usage
            tokens.update(ru.prompt_tokens, ru.completion_tokens)
            # pprint(ru)
            # pprint(tokens)
            print(f"prompt tokens: {ru.prompt_tokens}, completion tokens: {ru.completion_tokens}, total tokens: {ru.total_tokens} cost: {tokens.cost():.4f}")

    if len(history.messages) > 2:
        history.save(LOG_FILE)
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
    history.append(assistant_tool_message(tool_calls=[tc]))
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


if __name__ == "__main__":
    # test_message_history(); exit(0)
    parser = argparse.ArgumentParser(description="Chat with LLMs")
    parser.add_argument(
        "llm",
        choices=list(model_info.keys()),
        type=str,
        help="model key",
    )
    parser.add_argument("tool_use", type=str, nargs="?", default="", help="enable tool calls")

    args = parser.parse_args()
    chat(args.llm, args.tool_use == "tool")
