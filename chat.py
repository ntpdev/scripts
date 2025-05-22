import base64
import inspect
import json
import re
from datetime import datetime
from enum import StrEnum
from functools import cache
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

from openai import OpenAI
from openai.types.responses.response import Response
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from pydantic import BaseModel, Field, SerializeAsAny, field_validator
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint

import chatutils as cu

# ---
# chat using OpenAI responses API
# ---

console = Console()
client = OpenAI()
PYLINE_SPLIT = re.compile(r"; |\n")
role_to_color = {"system": "white", "developer": "white", "user": "green", "assistant": "cyan", "tool": "yellow"}

# types for message construction


class InputItem(BaseModel):
    type: Literal["input_text", "output_text", "input_image", "input_file"]


class InputText(InputItem):
    type: Literal["input_text"] = "input_text"
    text: str


class OutputText(InputItem):
    type: Literal["output_text"] = "output_text"
    text: str


class InputImage(InputItem):
    type: Literal["input_image"] = "input_image"
    image_url: str | Path

    @field_validator("image_url", mode="before")
    @classmethod
    def validate_image_url(cls, v: str | Path) -> str:
        if isinstance(v, Path):
            ext = v.suffix.lower().lstrip(".")
            fmt = "jpeg" if ext == "jpg" else ext
            console.print(f"uploading image {v}", style="yellow")
            data = v.read_bytes()
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:image/{fmt};base64,{b64}"
        # assume it’s already a data‑URI or URL string
        return v


class InputFile(InputItem):
    type: Literal["input_file"] = "input_file"
    file_id: str

    @field_validator("file_id", mode="before")
    @classmethod
    def validate_file_id(cls, v: str | Path) -> str:
        if isinstance(v, Path):
            xs = client.files.list(purpose="user_data")
            file = next((e for e in xs if e.filename == v.name), None)
            if not file:
                console.print(f"uploading file {v}", style="yellow")
                with v.open("rb") as fp:
                    file = client.files.create(file=fp, purpose="user_data")
            console.print(f"file {file.filename} {file.id}", style="yellow")
            return file.id
        # assume it is a valid id
        return v


class OpenAIModel(StrEnum):
    GPT = "gpt-4.1"
    GPT_MINI = "gpt-4.1-mini"
    O4 = "o4-mini"


class Role(StrEnum):
    ASSISTANT = "assistant"
    DEVELOPER = "developer"
    SYSTEM = "system"
    USER = "user"


class Message(BaseModel):
    role: Role
    content: list[SerializeAsAny[InputItem]]

    def __str__(self) -> str:
        # build and truncate content string, showing only text for input/output items
        parts: list[str] = []
        for item in self.content:
            text = getattr(item, "text", None)
            parts.append(text if text is not None else str(item))
        content_str = " ".join(parts)
        if len(content_str) > 70:
            content_str = content_str[:67] + "..."
        colour = role_to_color[self.role.value]
        return f"[{colour}]{self.role.value}: {content_str}[/{colour}]"


class FunctionCall(BaseModel):
    type: Literal["function_call"]
    id: str
    call_id: str
    name: str
    arguments: str

    def __str__(self) -> str:
        colour = role_to_color["tool"]
        return f"[{colour}]FunctionCall {self.name}({self.arguments})[/{colour}]"


class FunctionCallResponse(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str

    def __str__(self) -> str:
        colour = role_to_color["tool"]
        return f"[{colour}]FunctionCallResponse result = {self.output}[/{colour}]"


class ReasoningItem(BaseModel):
    id: str
    summary: list
    type: str

    def __str__(self) -> str:
        summary_str = " ".join(self.summary) if isinstance(self.summary, list) else str(self.summary)
        return f"ReasoningItem {self.id}: {summary_str}"


class MessageHistory(BaseModel):
    """tracks messages that have been sent to the LLM"""
    messages: list[Message | FunctionCall | FunctionCallResponse | ReasoningItem] = Field(default_factory=list)
    processed: list[bool] = Field(default_factory=list)

    def append(self, msg: Message | FunctionCall | FunctionCallResponse | ReasoningItem) -> None:
        self.messages.append(msg)
        # mark output_text messages as processed (True) because they do not need to be sent back
        self.processed.append((isinstance(msg, Message) and msg.content[0].type == "output_text") or isinstance(msg, FunctionCall) or isinstance(msg, ReasoningItem))

    def dump(self):
        result = []
        for i, msg in enumerate(self.messages):
            if not self.processed[i]:
                result.append(msg.model_dump())
                self.processed[i] = True
        return result

    def __repr__(self):
        return f"MessageHistory len={len(self.messages)} unprocessed={sum(not x for x in self.processed)}"

    def print(self) -> None:
        """Print the message history to console using each item's __str__."""
        console.print("\n".join(f"{i:2d} {m}" for i, m in enumerate(self.messages)))


# helper functions for message construction


def user_message(*items: str | InputItem) -> Message:
    return Message(role=Role.USER, content=[InputText(text=item) if isinstance(item, str) else item for item in items])


def developer_message(*items: str | InputItem) -> Message:
    return Message(role=Role.DEVELOPER, content=[InputText(text=item) if isinstance(item, str) else item for item in items])


def assistant_message(response: Response) -> Message:
    r, s, t = extract_text_content(response)
    return Message(role=r, content=[OutputText(type=t, text=s)])


def new_conversation(text: str) -> list[Message]:
    text += f"\nthe current date is {datetime.now().isoformat()}"
    return [developer_message(text)]


def function_call(tool_call: ResponseFunctionToolCall) -> FunctionCall:
    return FunctionCall(type=tool_call.type, id=tool_call.id, call_id=tool_call.call_id, name=tool_call.name, arguments=tool_call.arguments)


def function_call_response(tool_call: ResponseFunctionToolCall, result: Any) -> FunctionCallResponse:
    return FunctionCallResponse(call_id=tool_call.call_id, output=str(result))


def reasoning_item(item: ResponseReasoningItem) -> ReasoningItem:
    return ReasoningItem(id=item.id, summary=item.summary, type=item.type)


# structured output models


class Answer(BaseModel):
    number: int = Field(description="question number")
    choice: str = Field(description="the single word answer")


class AnswerSheet(BaseModel):
    answers: list[Answer] = Field(description="the list of answers")

    def to_yaml(self) -> str:
        lines = (f"  {x.number}: {x.choice}" for x in self.answers)
        return "answers:\n" + "\n".join(lines)


class Marked(BaseModel):
    number: int = Field(description="question number")
    answer: str = Field(description="given answer")
    expected: str = Field(description="correct answer")
    feedback: str = Field(description="explanation for incorrect answers")
    is_correct: bool = Field(description="True if correct")


class MarkSheet(BaseModel):
    answers: list[Marked] = Field(description="list of marked answers")
    correct: int = Field(description="count of correct answers")

    def to_yaml(self) -> str:
        xs = (f"  {x.number}:\n    answer: {x.answer}\n    correct: {x.expected}\n    mark: {'✓' if x.is_correct else '✘'}\n    feedback: {x.feedback}" for x in self.answers)
        joined = '\n'.join(xs)
        return f"answers:\n{joined}\nmark: {self.correct}"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def update(self, response_usage):
        self.input_tokens += response_usage.input_tokens
        self.output_tokens += response_usage.output_tokens
        self.reasoning_tokens += response_usage.output_tokens_details.reasoning_tokens


class LLM:
    """a stateful model which maintains a conversation thread using a response_id"""

    model: OpenAIModel
    instructions: str
    use_tools: bool
    response_id: str = None
    usage: Usage

    def __init__(self, model: OpenAIModel, instructions: str = "", use_tools: bool = False):
        self.model = model
        self.instructions = instructions
        self.use_tools = use_tools
        self.usage = Usage()

    def _create(self, args) -> Response:
        if self.response_id:
            args["previous_response_id"] = self.response_id
        response = client.responses.create(**args)
        self.response_id = response.id
        self.usage.update(response.usage)
        return response

    def create(self, history: MessageHistory) -> Response:
        """create a message based on existing conversation context. update history msg with assistant response"""

        args = {"model": self.model, "input": history.dump()}
        assert args["input"], "no input"
        if self.instructions:
            args["instructions"] = self.instructions
        is_reasoning = self.model.startswith("o")
        if is_reasoning:
            args["reasoning"] = {"effort": "low"}
            args["max_output_tokens"] = 8192
        else:
            args["max_output_tokens"] = 4096
        if self.use_tools:
            args["tools"] = [v["defn"] for v in fn_mapping().values()]
            if not is_reasoning:
                args["temperature"] = 0.6
        response = self._create(args)

        max_tool_calls = 9
        while max_tool_calls and any(e.type == "function_call" for e in response.output):
            process_function_calls(history, response)
            console.print(f"{10 - max_tool_calls}: returning function call results", style="yellow")
            # send results of function calls back to model
            args["input"] = history.dump()
            response = self._create(args)
            max_tool_calls -= 1

        if max_tool_calls == 0:
            console.print("tool call limit exceeded", style="red")

        history.append(assistant_message(response))
        return response


# uses the Responses API function defintions
eval_fn = {
    "type": "function",
    "name": "eval",
    "description": "evaluate a mathematical or Python expressions",
    "parameters": {
        "type": "object",
        "required": ["expression"],
        "properties": {"expression": {"type": "string", "description": "The expression"}},
        "additionalProperties": False,
    },
    "strict": True,
}

execute_script_fn = {
    "type": "function",
    "name": "execute_script",
    "description": "execute a PowerShell or Python script on the local computer. Include print statements to see output. Text written to stdout and stderr will be returned",
    "parameters": {
        "type": "object",
        "required": ["language", "script_lines"],
        "properties": {
            "language": {"type": "string", "enum": ["PowerShell", "Python"], "description": "the name of the scripting language either PowerShell or Python"},
            "script_lines": {"type": "array", "description": "The list of lines", "items": {"type": "string", "description": "a line"}},
        },
        "additionalProperties": False,
    },
    "strict": True,
}


create_tasklist_fn = {
    "type": "function",
    "name": "create_tasklist",
    "description": "create a new task list with all the steps that need to be done.",
    "parameters": {
        "type": "object",
        "required": ["tasks"],
        "properties": {
            "tasks": {"type": "array", "description": "The ordered list of tasks", "items": {"type": "string", "description": "a task"}},
        },
        "additionalProperties": False,
    },
    "strict": True,
}

mark_task_complete_fn = {
    "type": "function",
    "name": "mark_task_complete",
    "description": "mark a step on the active tasklist as complete. It will return the next task to be done.",
    "parameters": {
        "type": "object",
        "required": ["step"],
        "properties": {
            "step": {"type": "integer", "description": "Number of the task just completed"},
        },
        "additionalProperties": False,
    },
    "strict": True,
}

read_file_fn = {
    "type": "function",
    "name": "read_file",
    "description": "read the contents of a file on the local computer.",
    "parameters": {
        "type": "object",
        "required": ["filename", "line_number", "show_line_numbers", "window_size"],
        "properties": {
            "filename": {"type": "string", "description": "the file name"},
            "line_number": {"type": "integer", "description": "the center of the block of lines to show"},
            "show_line_numbers": {"type": "boolean", "description": "include line numbers in the output"},
            "window_size": {"type": "integer", "description": "number of lines to show before and after the line number"},
        },
        "additionalProperties": False,
    },
    "strict": True,
}

apply_diff_fn = {
    "type": "function",
    "name": "apply_diff",
    "description": dedent("""
        apply a diff to a file on the local computer. The diff must contains the line to be searched for in the file and the replacement lines. Example:
        <<<
        search
        ===
        replace_1
        replace_2
        >>>
     """),
    "parameters": {
        "type": "object",
        "required": ["filename", "diff"],
        "properties": {
            "filename": {"type": "string", "description": "the file name"},
            "diff": {"type": "string", "description": "the diff to apply"},
        },
        "additionalProperties": False,
    },
    "strict": True,
}


apply_unified_diff_fn = {
    "type": "function",
    "name": "apply_unified_diff",
    "description": dedent("""
        apply a unified diff to a file on the local computer. The diff should contain a single change affecting a block of lines. Do not add diff metadata. Example:

         line1
        -delete line2
         +new line
         +  indented line
         line3"""),
    "parameters": {
        "type": "object",
        "required": ["filename", "start_line", "diff"],
        "properties": {
            "filename": {"type": "string", "description": "the file name"},
            "start_line": {"type": "integer", "description": "the line number (1 based) to start the diff at"},
            "diff": {"type": "string", "description": "the diff to apply"},
        },
        "additionalProperties": False,
    },
    "strict": True,
}


# lifted from agents SDK
def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters with their descriptions.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {func.__name__}: {str(e)}")

    # Parameter descriptions - can be extended or replaced with docstring parsing
    param_descriptions = {"expression": "TODO"}

    parameters = {}
    required = []

    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")

            # Get description if available
            description = param_descriptions.get(param.name, "")

            parameters[param.name] = {"type": param_type, "description": description}

            # Add to required list if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param.name)

        except KeyError as e:
            raise KeyError(f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}")

    return {
        "type": "function",
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {
            "type": "object",
            "required": required,
            "properties": parameters,
            "additionalProperties": False,  # Assuming this is desired for all cases
        },
        "strict": True,  # Assuming this is desired for all cases
    }


@cache
def fn_mapping() -> dict[str, dict[str, Any]]:
    """Returns a dictionary mapping function names to their definitions and a callable."""
    return {
        eval_fn["name"].lower(): {"defn": eval_fn, "fn": evaluate_expression},
        execute_script_fn["name"].lower(): {"defn": execute_script_fn, "fn": execute_script},
        create_tasklist_fn["name"].lower(): {"defn": create_tasklist_fn, "fn": create_tasklist},
        mark_task_complete_fn["name"].lower(): {"defn": mark_task_complete_fn, "fn": mark_task_complete},
        read_file_fn["name"].lower(): {"defn": read_file_fn, "fn": read_file},
        # apply_diff_fn["name"].lower(): {"defn": apply_diff_fn, "fn": apply_diff},
        apply_unified_diff_fn["name"].lower(): {"defn": apply_unified_diff_fn, "fn": apply_simple_diff},
    }


def dispatch(fn_name: str, args: dict) -> Any:
    fn = fn_name.lower()
    if fn_entry := fn_mapping().get(fn):
        try:
            r = fn_entry["fn"](**args)
            if isinstance(r, str):
                if r.strip():
                    console.print(Markdown(r, style="yellow", code_theme="monokai"), width=80)
                else:
                    r = "SUCCESS: the tool executed but no output was returned. Add print statements."
                    console.print(r, style="red")
            return r
        except Exception as e:
            r = f"ERROR: tool call failed. {type(e).__name__} - {str(e)}"
            console.print(r, style="red")
            return r

    r = f'ERROR: No tool named "{fn_name}" found'
    console.print(r, style="red")
    return r


def process_function_call(tool_call: ResponseFunctionToolCall, history: MessageHistory) -> None:
    """process the tool call and append response to history"""
    console.print(f"{tool_call.type}: {tool_call.name} with {tool_call.arguments}", style="yellow")
    history.append(function_call(tool_call))
    # if we used parse then the arguments for the tool will have be extracted
    args = getattr(tool_call, "parsed_arguments", None)
    if not args:
        args = json.loads(tool_call.arguments)
    result = dispatch(tool_call.name, args)
    history.append(function_call_response(tool_call, result))


def process_function_calls(history: MessageHistory, response: Response) -> None:
    """execute all function calls and append to message history. echo reason items back"""
    message_printed = False
    for output in response.output:
        match output.type:
            case "function_call":
                process_function_call(output, history)
            case "reasoning":
                history.append(reasoning_item(output))  # echo the reasoning items back
            case "message":
                if not message_printed:
                    print_response(response)
                    message_printed = True  # Mark that we've printed
            case _:
                console.print(f"unexpected message type {output.type}", style="red")


def extract_text_block(p: Path) -> str:
    """Extract lines from the first code block in a markdown file."""
    in_block = False
    block_lines: list[str] = []

    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("```"):
                    if in_block:
                        return "".join(block_lines)
                    in_block = True
                elif in_block:
                    block_lines.append(line)
    except FileNotFoundError:
        return ""

    return ""


def print_message(msg) -> None:
    """print text if an instance of Message"""
    if isinstance(msg, Message):
        c = role_to_color[msg.role]
        s = "".join(e.text for e in msg.content if e.type in ["input_text", "output_text"])
        s = cu.translate_latex(s)
        console.print(Markdown(f"{msg.role}:\n{s}"), style=c)


def extract_text_content(response: Response) -> tuple[str, str, str]:
    """extract tuple of role, text, type from messages of type message / output_text"""
    output_text = ""
    role = ""
    content_type = ""
    for output in (x for x in response.output if x.type == "message"):
        role = output.role
        for item in (x for x in output.content if x.type == "output_text"):
            output_text += item.text
            content_type = item.type
    return Role(role), cu.translate_latex(output_text), content_type


def print_response(response: Response) -> tuple[str, str, str]:
    """extract tuple of role, text, type from messages of type message / output_text and print. does nothing if no text"""
    r, s, t = extract_text_content(response)
    if s:
        console.print(Markdown(f"{r} ({response.model}):\n\n{s}\n"), style=role_to_color[r])
    return r, s, t


def execute_script(language: str, script_lines: list[str]) -> Any:
    if len(script_lines) == 1:
        # llm might send multiline code as single escaped string
        s = script_lines[0].replace("\\n", "\n").replace("\\t", "\t")
        script_lines = s.splitlines()
    code = cu.CodeBlock(language.lower(), script_lines)
    return cu.execute_script(code)


def read_file(filename: str, line_number: int, show_line_numbers: bool = False, window_size: int = 50) -> list[str]:
    """
    Read lines from a file centered around a specified line number.
    
    Args:
        filename: Path to the file to read
        line_number: Center line number (1-based)
        show_line_numbers: Prefix lines with numbers
        window_size: Number of lines to return
    
    Returns:
        List of lines around the specified position
    """
    fn = Path("/code/scripts/" + filename)
    lines = fn.read_text(encoding='utf-8').splitlines()
    adjusted_pos = max(0, line_number - 1)  # Convert to 0-based index
    window_size = max(50, window_size)
    
    n = len(lines)
    if n <= window_size:
        start_idx, selected = 0, lines
    else:
        start_idx = max(0, min(adjusted_pos - window_size//2, n - window_size))
        selected = lines[start_idx:start_idx + window_size]
    
    console.print(f"read {fn.name} {show_line_numbers} lines {start_idx + 1} to {start_idx + len(selected) + 1}", style="yellow")
    if show_line_numbers:
        width = len(str(start_idx + window_size))
        return [f"{i:>{width}} {line}" for i, line in enumerate(selected, start=start_idx + 1)]
    
    return selected


def apply_diff(filename: str, diff: str) -> str:
    p = Path.home() / "Documents" / "chats" / filename
    console.print(f"reading file {p}", style="yellow")
    result = cu.apply_diff(p, diff)
    cu.print_block(result, True, style="white")
    p.write_text(result, encoding="utf-8")
    return "success"


def apply_simple_diff(filename: str, start_line: int, diff: str) -> str:
    cu.print_block(diff, True, style="yellow")
    p = Path("/code/scripts/" + filename)
    console.print(f"reading file {p}", style="yellow")
    result = cu.apply_simple_diff(p, start_line, diff)
    cu.print_block(result, True, style="white")
    p.write_text(result, encoding="utf-8")
    return "success"


def evaluate_expression_impl(expression: str) -> Any:
    # Split into individual parts removing blank lines but preserving indents
    parts = [e for e in PYLINE_SPLIT.split(expression) if e.strip()]
    if not parts:
        return None  # Empty input

    parts = ["import math", "import datetime"] + parts
    # Separate final expression
    *statements, last_part = parts

    # Create a namespace dictionary to store variables
    namespace = {}

    # Execute all statements updating the namespace as necessary
    if statements:
        exec("\n".join(statements), namespace)

    # Evaluate result of final expression
    return eval(last_part.strip(), namespace)


class TaskList:
    def __init__(self):
        self.tasks = []
        self.current_task_index = 0

    def create_tasklist(self, tasks: list[str]) -> str:
        self.tasks = [{"description": task, "completed": False} for task in tasks]
        self.current_task_index = 0
        return self._format_tasks()

    def mark_task_complete(self, step: int) -> str:
        if step < 1 or step > len(self.tasks):
            raise ValueError(f"Invalid task index {step} out of range 1 to {len(self.tasks)}")

        idx = step - 1

        if self.tasks[idx]["completed"]:
            return f"Task {step} is already completed."

        self.tasks[idx]["completed"] = True

        # Find the next incomplete task
        next_task_index = next((i for i, task in enumerate(self.tasks) if not task["completed"]), None)

        if next_task_index is None:
            return f"Subtask {step} completed. All tasks completed!"

        self.current_task_index = next_task_index
        return f"Subtask {step} completed.\nThe next subtask is\n{next_task_index + 1} {self.tasks[next_task_index]['description']}"

    def _format_tasks(self) -> str:
        """
        Formats the tasks as a numbered list in markdown.

        Returns:
        str: The formatted tasklist.
        """
        xs = []
        for i, task in enumerate(self.tasks):
            status = "☑" if task["completed"] else "☐"
            task_line = f"{i + 1}. {task['description']} {status}"
            xs.append(task_line)

        return "current tasks:\n" + "\n".join(xs)


tasklist: TaskList | None = None


def create_tasklist(tasks: list[str]) -> str:
    global tasklist
    tasklist = TaskList()
    return tasklist.create_tasklist(tasks)


def mark_task_complete(step: int) -> str:
    return tasklist.mark_task_complete(step)


def evaluate_expression(expression: str) -> Any:
    result = ""
    if expression:
        console.print("eval: " + expression, style="yellow")
        result = evaluate_expression_impl(expression)
        console.print("result: " + str(result), style="yellow")
    else:
        result = "ERROR: expression is empty"
        console.print(result, style="red")
    return result


def simple_message():
    """a simple multi-turn conversation maintaining coversation state using response.id"""
    prompts = [
        "list the last 3 UK prime ministers give the month and year they became PM. Who is PM today?",
        "does the answer correctly express uncertainty. When will the next UK general election be held.",
        "review your last answers and check for logical and factual consistency. are you confident that Rishi Sunak is UK Prime Minister today",
    ]

    dev_inst = f"The assistant is Marvin an AI chatbot. The assistant gives concise answers and expresses uncertainty when needed. The current date is {datetime.now().isoformat()}"
    # models can confidently say that Rishi Sunak is PM in 2025
    console.print(Markdown(dev_inst), style="white")
    response_id = None
    for prompt in prompts:
        console.print(Markdown(f"user:\n\n{prompt}\n"), style="green")
        response = client.responses.create(
            model=OpenAIModel.GPT,
            instructions=dev_inst,
            previous_response_id=response_id,
            input=prompt,
        )
        console.print(Markdown(f"assistant ({response.model}):\n\n{response.output_text}\n"), style="cyan")
        response_id = response.id


def test_search_example():
    question = cu.load_textfile("search1.md")
    answer = cu.load_textfile("search1-ans.md")
    dev_inst = f"You are Marvin an AI chatbot who gives insightful and concise answers to questions which are always based on evidence. The current date is {datetime.now().isoformat()}"

    response_id = None
    for prompt in [question, answer]:
        console.print(Markdown(f"user:\n\n{prompt}\n"), style="green")
        response = client.responses.create(
            model=OpenAIModel.GPT_MINI,
            instructions=dev_inst,
            previous_response_id=response_id,
            input=prompt,
        )
        console.print(Markdown(f"assistant ({response.model}):\n\n{response.output_text}\n"), style="cyan")
        response_id = response.id


def test_file_inputs():
    """upload a pdf and ask questions about the content"""
    llm = LLM(OpenAIModel.GPT_MINI, instructions="role: AI researcher")
    history = MessageHistory()

    questions = dedent("""\
        answer the following questions based on information in the file.
        1. what games are discussed
        2. what competition is mentioned
        3. is a chess player mentioned
        4. what was the name of the computer chess program
        """)
    pdf = Path.home() / "Downloads" / "bitter_lesson.pdf"
    msg = user_message(questions, InputFile(file_id=pdf))
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)

    msg = user_message("Summarise the thesis put forward. How relevant is it in 2024?")
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)


def test_function_calling():
    sys_msg = dedent("""\
        You have access to the eval tool. use the eval tool to evaluate any Python expression. Use it for more complex mathematical operations, counting, searching, sorting and date calculations.
        examples of using eval are:
        - user what is "math.pi * math.sqrt(2)" -> 4.442882938158366
        - user what are the first 5 cubes "[x ** 3 for x in range(1,6)]" -> [1, 8, 27, 64, 125]
        - user how many times does l occur in hello world" 'hello world'.count('l')" -> 3
        - user how many days between 4th March and Christmas in 2024 "datetime.date(2024,12,25) - datetime.date(2024,3,4)).days" -> 296
        - user what is "1 + 2 * 3" -> no need to use eval the answer is 7
        """)
    llm = LLM(OpenAIModel.GPT_MINI, "", True)
    history = MessageHistory()
    dev_msg = developer_message(sys_msg)
    history.append(dev_msg)

    prompts = [
        "a circle is inscribed in a square. the square has a side length of 3 m. what is the area inside the square but not in the circle.",
        r"find the roots of \(2x^{2}-5x-6\). express to 3 dp",
        "how many days since the first moon landing",
        "which has more letter 'r' in it Strawberry or Raspberry",
    ]
    for prompt in prompts:
        msg = user_message(prompt)
        print_message(msg)
        history.append(msg)
        response = llm.create(history)
        print_response(response)
    history.print()


def test_function_calling_python():
    sys_msg = dedent(f"""\
        You are Marvin an AI assistant. You have access to two tools:
        - eval: use the eval tool to evaluate any Python expression. Use it for more complex mathematical operations, counting, searching, sorting and date calculations.
        - execute_script: use execute_script tool to run a PowerShell script or Python program on the local computer. Remeber to add print statements to Python code. The output from stdout and stderr will be return to you.        
         the current datetime is {datetime.now().isoformat()}
        """)
    llm = LLM(OpenAIModel.GPT_MINI, "analyse the complexity of the problem to determine whether to use manual calculation or a tool", True)
    history = MessageHistory()
    history.append(developer_message(sys_msg))
    msg = user_message("is 30907 prime. if not what are its prime factors")
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)

    question = dedent("""\
        The 9 members of a baseball team went to an ice-cream parlor after their game.
        Each player had a single-scoop cone of chocolate, vanilla, or strawberry ice cream.
        At least one player chose each flavor, and the number of players who chose chocolate was greater than the number of players who chose vanilla, which was greater than the number of players who chose strawberry.
        Let N be the number of different assignments of flavors to players that meet these conditions.
        Find the remainder when N is divided by 1000.
        """)
    msg = user_message(question)
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)


def test_function_calling_powershell():
    dev_inst = dedent(f"""\
        You are Marvin an AI assistant.
        You have access to two tools:
        - eval: use the eval tool to evaluate any Python expression. Use it for more complex mathematical operations, counting, searching, sorting and date calculations.
        - execute_script: use execute_script tool to run a PowerShell script or Python program on the local computer. Remeber to add print statements to Python code. The output from stdout and stderr will be return to you.
        the current datetime is {datetime.now().isoformat()}
        """)
    llm = LLM(OpenAIModel.GPT_MINI, "", True)
    history = MessageHistory()
    dev_msg = developer_message(dev_inst)
    history.append(dev_msg)

    question = dedent("""\
        ## task
        1. check files in ~/Documents/chats matching *.md and have robot in the file name
        2. find smallest such file
        3. read contents and summarise
        4. comment on whether the arguments are consistent with current state-of-the-art robotics in 2024. How reasonable are the predictions
        """)
    msg = user_message(question)
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)

    question = "look in ~/Documents/chats - find file prices-a.md . read file and answer question"
    msg = user_message(question)
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)


def test_git_workflow():
    dev_inst = dedent(f"""\
        You are Marvin an AI assistant. You are an expert using Git version control. 
        You have access to 4 tools:
        - eval: use the eval tool to evaluate a mathematical or Python expression. Use it for more complex mathematical operations, counting, searching, sorting and date calculations.
        - execute_script: use execute_script tool to run a PowerShell script or Python program on the local computer. Remeber to add print statements to Python code. The output from stdout and stderr will be return to you.
        - create_tasklist: use the create_tasklist tool to create an ordered list of steps that need to be completed. use this to keep track of multi-step plans
        - mark_task_complete: use the mark_task_complete tool to mark a step as complete and get a reminder of the next step
        the current datetime is {datetime.now().isoformat()}
        """)
    llm = LLM(OpenAIModel.GPT, "", True)
    # llm = LLM(OpenAIModel.GPT_MINI, "use the eval tool as needed", True)
    history = MessageHistory()
    dev_msg = developer_message(dev_inst)
    history.append(dev_msg)

    prompt = "## task\ncommit the modified files with the message 'updates by Marvin'\n\n## instructions\nfirst write out a plan and confirm with user.\nif the user agrees complete the entire plan and summarise outcome. Use powershell to execute git commands"
    msg = user_message(prompt)
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)

    prompt = "proceed"
    msg = user_message(prompt)
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)


def test_code_edit():
    dev_inst = dedent(f"""\
        The assistant is Marvin an expert Python programmer. 
        You have access to 2 tools:
        - read_file: use the read_file tool to read a file.
        - apply_diff: use apply_diff tool to makes edits to a file. each edit is represented as a diff block. 

        ## diff format
        each diff block contains two sections the lines to be searched for and the replacement lines. use <<< === >>> delimiters to mark the sections of the diff block. 
        minimise the size of the diff blocks by only including the lines that need to be changed and one or two lines for context.

        <<<
        line 1
        line 2
        ===
        line 1
        modified line 2
        new line 3
        >>>

        the current datetime is {datetime.now().isoformat()}
        """)
    llm = LLM(OpenAIModel.O4, "", True)
    history = MessageHistory()
    dev_msg = developer_message(dev_inst)
    history.append(dev_msg)
    fname = "temp.py"
    fn = cu.make_fullpath(fname)
    
    prompt = dedent(f"""\
        ## task
        modify the Python code in the file {fn.name}. The code should target Python 3.12
        
        ## instructions
        - implement the function extract_diff
        - do not run any code
        """)
    code = cu.load_textfile(fname)
    prompt += "\n\n" + code
    msg = user_message(prompt)
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)

    prompt = cu.run_python_unittest(fn)
    msg = user_message(prompt)
    print_message(msg)
    if "failed" in prompt.lower():
        history.append(msg)
        response = llm.create(history)
        print_response(response)

        # - add type hints to functions. use python 3.10 e.g. list | tuple and types from collections.abc
        # - current_diff is a dictionary of (search, replace) line tuples. replace with a dataclass
        # - summarise the changes for the user. do not repeat the code
        # - explain the purpose and logic of the code

def test_code_edit_unified():
    dev_inst = dedent(f"""\
        The assistant is Marvin an expert Python programmer. 
        You have access to 2 tools:
        - read_file: use the read_file tool to read a file.
        - apply_unified_diff: use apply_unified_diff tool to makes edits to a file. 

        the current datetime is {datetime.now().isoformat()}
        """)
    llm = LLM(OpenAIModel.O4, "", True)
    history = MessageHistory()
    dev_msg = developer_message(dev_inst)
    history.append(dev_msg)
    fname = Path("/code/scripts/chat.py")
    s = cu.run_linter(fname)
    if "error" in s.lower():  
        prompt = dedent(f"""\
            ## task
            read the output of the lint tool and fix {fname.name}
            
            ## instructions
            - understand the lint output and make the necessary changes to the code
            - show the unified diff to the user
            - make the smallest changes possible to fix the lint issues
            """)
        prompt += f"\n##lint output\nruff check --fix {fname.name}\n\n" + s
        msg = user_message(prompt)
        print_message(msg)
        history.append(msg)
        response = llm.create(history)
        print_response(response)


def test_image_analysis():
    """test image analysis of web image and local uploaded image. Use flex tier with higher timeouts. only for o4 models"""
    mdl = "o4-mini"
    cl = client.with_options(timeout=900.0)
    msg = user_message(InputText(text="describe the scene and the outfit"), InputImage(image_url="https://i.dailymail.co.uk/1s/2024/06/08/22/85884439-0-image-a-144_1717882540270.jpg"))
    response = cl.responses.create(model=mdl, input=[msg.model_dump()], service_tier="flex")
    print_response(response)
    pprint(response.usage)

    msg = user_message(InputText(text="extract the paragraph of text and put it inside a markdown block enclosed with <text> tags. describe the balances scales in this drawing"), InputImage(image_url=Path.home() / "Downloads" / "sum1.png"))
    response = cl.responses.create(model=mdl, input=[msg.model_dump()], service_tier="flex")
    print_response(response)
    pprint(response.usage)


def test_solve_visual_maths_problem():
    llm = LLM(OpenAIModel.GPT_MINI)
    history = MessageHistory()
    msg = user_message(InputText(text="extract the paragraph of text and put inside a markdown block enclosed with <text> tags. describe the balances scales in this drawing"), InputImage(image_url=Path.home() / "Downloads" / "sum1.png"))
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)

    msg = user_message(InputText(text="answer the question."))
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)

    msg = user_message(
        dedent("""\
            Task: mark the previous answer against the model answer.
            Award 1 mark for every correct inequality and 1 mark for every correct weight.
            Model answer: The 4 inequalities are S > C, T > 2C, 2S > T + C, 2C > S. Given all are natural numbers less than 10, the only solution is C=4, S=7, T=9
            """)
    )
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)


questions_q5 = """\
## question
for each question choose a single word from the Choices list that has the most similar meaning to words in lists A and B.

```yaml
{input}
```
"""

answers_q5 = """\
## task
check the student answers against this list of correct answers. mark each answer. For wrong answers provide feedback explaining why the correct word is a better fit. no comment is required for correct answers. give the final score.

## correct answers
```yaml
{answers}
```

## student answers
{input}
"""

questions_q6 = """\
## task
for each question choose the word from the choices list that means the same or nearly the same a the first word

```yaml
{input}
```
"""

answers_q6 = """\
## task
check the previous answers against this list of correct answers. mark each answers. For wrong answers provide feedback on why it is not the best choice. no comment is required for correct answers. give the final score.

## correct answers
```yaml
{answers}
```

## student answers
{input}
"""


def structured_output_message():
    """Answers question returning json output then reformat to yaml and get it marked returning json. The answer model sometimes marks the answers out of order doing the wrong ones first"""

    def load_and_insert_into_template(fn: Path, template: str, placeholder: str) -> str:
        return template.replace(placeholder, extract_text_block(fn))

    def process_response_output(response_outputs):
        ans = ""
        # ignore output of type ResponseReasoningItem which are output by reasoning models
        for output in (x for x in response_outputs if x.type == "message"):
            for item in output.content:
                # the raw JSON is available in item.text but will use the parsed object
                # parsed only exists if the return type is ParsedResponseOutputText
                if parsed := getattr(item, "parsed", None):
                    console.print("\nresponse JSON converted to YAML\n")
                    ans = f"```yaml\n{parsed.to_yaml()}\n```"
                    console.print(Markdown(ans), style="cyan")
                else:
                    console.print("ERROR: no parsed output returned", style="red")
        return ans

    def ask_question_and_mark(question: str, answer_template: str) -> None:
        console.print(Markdown(question), style="green")

        # First parse: retrieve response from model "o4-mini" with low reasoning effort
        # response_first = client.responses.parse(model="o4-mini", reasoning={"effort": "low"}, input=[user_message(InputText(text=question)).model_dump()], text_format=AnswerSheet)
        response_first = client.responses.parse(model=OpenAIModel.GPT, input=[user_message(InputText(text=question)).model_dump()], text_format=AnswerSheet)

        # Convert returned json to YAML format using process_response_output
        yaml_ans = process_response_output(response_first.output)

        # Replace the placeholder "{input}" in the answer template with the YAML output
        inp = answer_template.replace("{input}", yaml_ans)
        console.print(Markdown(inp), style="green")

        # Second parse: pass the updated answer input to OpenAIModel.GPT_MINI model with MarkSheet format
        response_final = client.responses.parse(model=OpenAIModel.GPT_MINI, input=[user_message(InputText(text=inp)).model_dump()], text_format=MarkSheet)

        # Process and print the final output
        process_response_output(response_final.output)

    root = Path.home() / "Documents" / "chats"
    # extract the questions from original and insert into new template to allow different instructions
    q5 = load_and_insert_into_template(root / "q5.md", questions_q5, "{input}")
    q5_ans = load_and_insert_into_template(root / "q5-ans.md", answers_q5, "{answers}")
    ask_question_and_mark(q5, q5_ans)

    # Process q6
    q6 = load_and_insert_into_template(root / "q6.md", questions_q6, "{input}")
    q6_ans = load_and_insert_into_template(root / "q6-ans.md", answers_q6, "{answers}")
    ask_question_and_mark(q6, q6_ans)


sysmsg = """\
You are Marvin, an expert analytical assistant. Communicate with precision, depth, and intellectual rigour.

Core Principles
- Prioritize accuracy and critical analysis over consensus viewpoints or popular narratives
- Adapt technical depth to match the user's demonstrated expertise
- Present nuanced perspectives rather than simplified binary choices
- Express appropriate uncertainty and distinguish between facts, expert opinions, and speculation

Communication Guidelines
1. Language Precision:
   - Use domain-specific terminology where appropriate
   - Define specialized terms only when context suggests unfamiliarity
   - Favour specific, concrete language over vague generalizations

2. Structural Clarity:
   - Organize complex responses with clear headers
   - Begin with key insights before elaborating
   - Use bold formatting **sparingly** to highlight crucial concepts

3. Intellectual Honesty:
   - Explicitly state your reasoning process and assumptions
   - Acknowledge limitations in available information
   - Present competing perspectives fairly, articulating the strongest version of each position

4. Response Quality:
   - Balance conciseness with thoroughness
   - Prioritize insight density over word count
   - Avoid rhetorical filler and empty phrasing

When discussing contentious topics, go beyond media talking points to examine underlying assumptions, methodological considerations, and contextual factors that shape different interpretations of the evidence.
"""


def test_chat_loop():
    """a simple multi-turn conversation maintaining coversation state using response.id"""

    def input_multi_line() -> str:
        if (inp := input().strip()) != "{":
            return inp
        lines = []
        while (line := input()) != "}":
            lines.append(line)
        return "\n".join(lines)

    dev_inst = f"The assistant is Marvin an AI chatbot. The assistant uses precise language and avoids vague or generic statements. The current date is {datetime.now().isoformat()}"
    dev_inst = dedent(f"""\
        The assistant is Marvin a helpful AI chatbot. Marvin is an expert python programmer.
        task: help the user write a design. the user will provides requirements and you should use those to formulate a design
        you will need to collaborate with the user to refine the design until the user is satisfied
        keep a track of all important decisions and the reasoning behind them
        The current date is {datetime.now().isoformat()}
        """)
    model = LLM(OpenAIModel.GPT, use_tools=True)
    history = MessageHistory()
    history.append(developer_message(dev_inst))
    attachments = []
    history.print()
    inp = ""
    while True:
        inp = input_multi_line()
        if inp == "x":
            break
        if inp.startswith("%attach"):
            if s := cu.load_textfile(inp.split()[1]):
                attachments.append(s)
            continue

        for a in attachments:
            inp += "\n" + a
        console.print(Markdown(f"user:\n\n{inp}\n"), style="green")
        history.append(user_message(inp))
        response = model.create(history)
        console.print(Markdown(f"assistant ({response.model}):\n\n{response.output_text}\n"), style="cyan")
        attachments.clear()

    pprint(model.usage)
    history.print()
    # pprint(history)


def main():
    # simple_message()
    # test_search_example()
    # test_file_inputs()
    # test_function_calling()
    # test_function_calling_python()
    # test_function_calling_powershell()
    # test_image_analysis()
    # test_solve_visual_maths_problem()
    # structured_output_message()
    test_git_workflow()
    # test_code_edit()
    # test_code_edit_unified()
    # test_chat_loop()



if __name__ == "__main__":
    main()
