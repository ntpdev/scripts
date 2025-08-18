import base64
import inspect
import json
import re
import sys
from datetime import datetime
from enum import StrEnum
from functools import cache
from pathlib import Path
from textwrap import dedent, shorten
from typing import Any, Literal, TypedDict

from openai import OpenAI
from openai.types.responses.response import Response
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from pydantic import BaseModel, Field, SerializeAsAny, field_validator
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint

import chatutils as cu
import toolutils as tu


# TypedDict for OpenAI function definitions
class FunctionDef(TypedDict, total=False):
    type: Literal["function"]
    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool


# ---
# chat using OpenAI responses API
# ---

console = Console()
client = OpenAI()
vfs = tu.VirtualFileSystem()
PYLINE_SPLIT = re.compile(r"; |\n")
IS_LINUX = sys.platform.startswith("linux")
role_to_color = {"system": "white", "developer": "white", "user": "green", "assistant": "cyan", "tool": "yellow"}
# external_file_map: dict[str, Path] = {}  # controls which files are visible to the LLM

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
    GPT = "gpt-5"
    GPT_MINI = "gpt-5-mini"
    O4 = "o4-mini"


# Costs are per million tokens (input / output)
PRICING = {
    OpenAIModel.GPT: (1.25, 10.00),
    OpenAIModel.GPT_MINI: (0.25, 2.00),
    OpenAIModel.O4: (1.10, 4.40),
}


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
        content_str = shorten(content_str, 70, placeholder="...")
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
        joined = "\n".join(xs)
        return f"answers:\n{joined}\nmark: {self.correct}"


class Usage(BaseModel):
    model: OpenAIModel
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def update(self, response_usage):
        self.input_tokens += response_usage.input_tokens
        self.output_tokens += response_usage.output_tokens
        self.reasoning_tokens += response_usage.output_tokens_details.reasoning_tokens

    def calculate_cost(self) -> float:
        """Calculates the cost based on accumulated usage and the stored model pricing."""
        if self.model not in PRICING:
            return 0.0

        input_price_per_million, output_price_per_million = PRICING[self.model]

        input_cost = (self.input_tokens / 1_000_000) * input_price_per_million

        total_output_tokens = self.output_tokens + self.reasoning_tokens
        output_cost = (total_output_tokens / 1_000_000) * output_price_per_million

        return input_cost + output_cost

    def __str__(self) -> str:
        return f"Usage:\n  Input Tokens: {self.input_tokens}\n  Output Tokens: {self.output_tokens}\n  Reasoning Tokens: {self.reasoning_tokens}\n  Estimated Cost: ${self.calculate_cost():.5f}"


class LLM:
    """a stateful model which maintains a conversation thread using the response_id"""

    model: OpenAIModel
    instructions: str
    use_tools: bool
    response_id: str | None = None
    usage: Usage

    def __init__(self, model: OpenAIModel, instructions: str = "", use_tools: bool = False):
        self.model = model
        self.instructions = instructions
        self.use_tools = use_tools
        self.usage = Usage(model=model)

    def _create(self, args) -> Response:
        if self.response_id:
            args["previous_response_id"] = self.response_id
        response = client.responses.create(**args)
        self.response_id = response.id
        self.usage.update(response.usage)
        return response

    def create(self, history: MessageHistory) -> Response:
        """create a message based on existing conversation context. update history msg with assistant response"""

        def is_tool_call(e):
            return e.type in ["function_call", "custom_tool_call"]

        args = {"model": self.model, "input": history.dump()}
        assert args["input"], "no input"
        if self.instructions:
            args["instructions"] = self.instructions
        is_reasoning = self.model.startswith("o")
        if is_reasoning:
            args["reasoning"] = {"effort": "low"}
            args["max_output_tokens"] = 8192
        else:
            # GPT-5 allow minimal reasoning and verbosity for agentic tasks
            args["reasoning"] = {"effort": "medium"}
            args["text"] = {"verbosity": "low"}
            args["max_output_tokens"] = 4096
        if self.use_tools:
            args["tools"] = [v["defn"] for v in fn_mapping().values()]
            # gpt-5 models do not support temperature or top-p
            # if not is_reasoning:
            #     args["temperature"] = 0.6
        response = self._create(args)

        max_tool_calls = 15
        while max_tool_calls and any(is_tool_call(e) for e in response.output):
            process_function_calls(history, response)
            console.print(f"{15 - max_tool_calls}: returning function call results", style="yellow")
            # send results of function calls back to model
            args["input"] = history.dump()
            response = self._create(args)
            max_tool_calls -= 1

        if max_tool_calls == 0:
            console.print("tool call limit exceeded", style="red")
        history.append(assistant_message(response))
        return response


# uses the Responses API function defintions
eval_fn: FunctionDef = {
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

execute_script_fn: FunctionDef = {
    "type": "function",
    "name": "execute_script",
    "description": "execute a Bash, PowerShell or Python script on the local computer. For python code add print statements to see output. Text written to stdout and stderr will be returned",
    "parameters": {
        "type": "object",
        "required": ["language", "script_lines"],
        "properties": {
            "language": {"type": "string", "enum": ["Bash", "PowerShell", "Python"], "description": "the name of the scripting language either Bash or PowerShell or Python"},
            "script_lines": {"type": "array", "description": "The list of lines", "items": {"type": "string", "description": "a line"}},
        },
        "additionalProperties": False,
    },
    "strict": True,
}


create_tasklist_fn: FunctionDef = {
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

mark_task_complete_fn: FunctionDef = {
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


apply_diff_fn: FunctionDef = {
    "type": "function",
    "name": "apply_diff",
    "description": "Applies a sequence of diff hunks to a text file. Diffs are applied in order. Provide at least 1 unchanged context line in each diff",
    "strict": True,
    "parameters": {
        "type": "object",
        "required": ["filename", "diffs"],
        "properties": {
            "filename": {"type": "string", "description": "The name of the file to which diffs will be applied"},
            "diffs": {
                "type": "array",
                "description": "Ordered list of diffs (hunks) to apply",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_position": {"type": "integer", "description": "1‐based index in the file where this diff is applied."},
                        "diff": {"type": "string", "description": "Unified diff format. prefix lines with ' ' unchanged '-' delete '+' insert."},
                    },
                    "required": ["start_position", "diff"],
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    },
}

read_text_fn: FunctionDef = {
    "type": "function",
    "name": "read_text",
    "description": "Read a window of lines from a file centered around a specified line number.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "a file name"},
            "line_number": {"type": "integer", "description": "1-based center line number"},
            "window_size": {"type": "integer", "description": "Number of lines to return"},
            "show_line_numbers": {"type": "boolean", "description": "Whether to prefix lines with their line numbers"},
        },
        "required": ["filename", "line_number", "window_size", "show_line_numbers"],
        "additionalProperties": False,
    },
}

apply_edits_fn: FunctionDef = {
    "type": "function",
    "name": "apply_edits",
    "description": "apply a series of edits defined as search-replace blocks. Each block is delimited by `<<<<<<<` `=======`  `>>>>>>> .",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {"filename": {"type": "string", "description": "a filename"}, "edits": {"type": "string", "description": "A multi-line string containing 1 or more search-replace blocks."}},
        "required": ["filename", "edits"],
        "additionalProperties": False,
    },
}

edit_text_fn = {"type": "custom", "name": "apply_edits", "description": "Applies one or more changes to a text file. Write as many changes as necessary. Format each change as a search replace block using the delimiters ---search ---replace ---end"}


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
        read_text_fn["name"].lower(): {"defn": read_text_fn, "fn": read_text},
        # apply_edits_fn["name"].lower(): {"defn": apply_edits_fn, "fn": apply_edits},
        # edit_text_fn["name"].lower(): {"defn": edit_text_fn, "fn": edit_text},
        # apply_diff_fn["name"].lower(): {"defn": apply_diff_fn, "fn": apply_diff},
    }


def dispatch(fn_name: str, args: dict) -> Any:
    fn = fn_name.lower()
    if fn_entry := fn_mapping().get(fn):
        try:
            r = fn_entry["fn"](**args)
            if isinstance(r, str):
                if r.strip():
                    console.print(r, style="yellow", markup=False)
                else:
                    r = "success: the tool executed but no output was returned. Add print statements."
                    console.print(r, style="red")
            return r
        except Exception as e:
            r = f"error: tool call failed. {type(e).__name__} - {str(e)}"
            console.print(r, style="red")
            return r

    r = f'error: No tool named "{fn_name}" found'
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


def process_custom_tool_call(tool_call: ResponseOutputMessage, history: MessageHistory) -> None:
    """process the custom tool call and append response to history"""
    input = getattr(tool_call, "input", "")
    console.print(f"{tool_call.type}: {tool_call.name} with {len(input)} chars", style="yellow")
    result = dispatch(tool_call.name, {"input": input})
    history.append(function_call_response(tool_call, result))


def process_function_calls(history: MessageHistory, response: Response) -> None:
    """execute all function calls and append to message history. echo reason items back"""
    message_printed = False
    for output in response.output:
        match output.type:
            case "custom_tool_call":
                process_custom_tool_call(output, history)
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


def read_text(filename: str, line_number: int = 1, window_size: int = 50, show_line_numbers: bool = False) -> list[str]:
    """
    Read lines from a file centered around a specified line number.

    Parameters:
        filename: Path to the file to read
        line_number: Center line number (1-based)
        show_line_numbers: Prefix lines with numbers
        window_size: Number of lines to return

    Returns:
        List of lines around the specified position
    """
    try:
        return vfs.read_text(filename, line_number, window_size, show_line_numbers)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def apply_edits(filename: str, edits: str) -> str:
    """
    Apply search-and-replace edits to the contents of a file.

    The `edits` string should contain one or more edit blocks in the following format:

        --- search
        line1_to_search
        line2_to_search
        --- replace
        replacement_line1
        replacement_line2
        --- end

    For each edit section:
      The block of text in the file that exactly matches the lines
      between `--- search` and `--- replace` is replaced by
      the block with the lines between `--- replace` and `--- end`.

    Parameters:
        filename (str):
            Path to the file whose contents will be edited.
        edits (str):
            Multiline string containing one or more edit blocks. Each block
            is delimited by the markers `--- search`, `--- replace` and `--- end`.

    Returns:
        str:
            SUCCESS or ERROR.
    """
    try:
        return vfs.apply_edits(filename, cu.CodeBlock(language="", lines=edits.splitlines))
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def edit_text(input: str) -> str:
    """
    Custom tool implementation
    """
    cu.print_block(input, line_numbers=True)
    return "SUCCESS"


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
    prompts = ["list the last 3 UK prime ministers give the month and year they became PM. Who is PM today?",
                "which of these were elected."]

    dev_inst = f"The assistant is Marvin an AI chatbot. Give concise answers and expresses uncertainty when needed. The current date is {datetime.now().isoformat()}"
    console.print(Markdown(dev_inst), style="white")
    response_id = None
    for prompt in prompts:
        console.print(Markdown(f"user:\n\n{prompt}\n"), style="green")
        response = client.responses.create(
            model=OpenAIModel.GPT_MINI,
            instructions=dev_inst,
            previous_response_id=response_id,
            input=prompt,
        )
        console.print(Markdown(f"assistant ({response.model}):\n\n{response.output_text}\n"), style="cyan")
        response_id = response.id


def test_search_example():
    question = cu.load_textfile(cu.make_fullpath("search1.md"))
    answer = cu.load_textfile(cu.make_fullpath("search1-ans.md"))
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
        "find the roots of x^3 - 2x^2 -8x - 35 = 0. Show real and complex roots. express to 3 dp",
        "how many days since the first moon landing",
        "which word has more letters 'r' in it - Strawberry or Raspberry",
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


def test_function_calling_shell():
    lang = "Bash" if IS_LINUX else "Powershell"
    dev_inst = dedent(f"""\
        You are Marvin an AI assistant.
        You have access to two tools:
        - eval: use the eval tool to evaluate any Python expression. Use it for more complex mathematical operations, counting, searching, sorting and date calculations.
        - execute_script: use execute_script tool to run a {lang} script or Python program on the local computer. Remeber to add print statements to Python code. The output from stdout and stderr will be return to you.
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


def git_workflow():
    shell = "bash" if IS_LINUX else "powershell"
    dev_inst = dedent(f"""\
        The assistant is Marvin an expert at using Git version control.
        You are an agent. Keep going until the task is completed, before ending your turn.
        You have access to 4 tools:
        - eval: use the eval tool to evaluate a mathematical or Python expression. Use it for more complex mathematical operations, counting, searching, sorting and date calculations.
        - execute_script: use execute_script tool to run a {shell} script or Python program on the local computer. Remember to add print statements to Python code. The output from stdout and stderr will be return to you.
        - create_tasklist: use the create_tasklist tool to create an ordered list of steps that need to be completed. use this to keep track of multi-step plans
        - mark_task_complete: use the mark_task_complete tool to mark a step as complete and get a reminder of the next step
        the current datetime is {datetime.now().isoformat()}
        """)
    llm = LLM(OpenAIModel.GPT_MINI, "", True)
    history = MessageHistory()
    dev_msg = developer_message(dev_inst)
    history.append(dev_msg)

    p = "~/code/scripts" if IS_LINUX else "/code/scripts"
    prompt = dedent(f"""\
        ## task
        commit the modified files in the directory {p} with the message 'updates by Marvin'. Do not commit any untracked files.
        
        ## instructions
        first write out a plan and confirm with user.
        if the user agrees complete the entire plan and summarise outcome. Use {shell} to execute git commands""")
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
    console.print(llm.usage)


def test_apply_diff():
    dev_inst = dedent(f"""\
        The assistant is Marvin an expert Python programmer. 
        You have access to 2 tools:
        - read_file: use the read_file tool to read a file.
        - apply_diff: use apply_diff tool to makes edits to sections of a text file line by line.

        the current datetime is {datetime.now().isoformat()}
        """)
    llm = LLM(OpenAIModel.O4, "", True)
    history = MessageHistory()
    dev_msg = developer_message(dev_inst)
    history.append(dev_msg)
    fname = "test-apply-diff.md"

    prompt = cu.load_textfile(cu.make_fullpath(fname))
    msg = user_message(prompt)
    print_message(msg)
    history.append(msg)
    response = llm.create(history)
    print_response(response)
    console.print(llm.usage)


def lint_workflow(fname: Path):
    vfs.create_mapping(fname)
    success, lint_output = cu.run_linter(fname)
    if success:
        return

    dev_inst = dedent(f"""\
        The assistant is a senior Python developer.
        You are an agent. Keep going until the task is completed, before ending your turn.
        You have access to 1 tool:
        - read_file: use the read_file tool to read a file.

        the current datetime is {datetime.now().isoformat()}
        """)
    msg = dedent(f"""\
        task: fix lint errors

        instructions:
        - target Python version 3.12
        - understand the lint errors
        - edit the file to fix the lint errors
        - minimize the number of individual edits by grouping changes to nearby lines into a single operation
        - do not make any functional changes

        ## How to write changes

        Output structured edits in this exact format:

        ```
        --- search filename
        context line 1
        line 2
        --- replace
        context line 1
        new line
        --- end
        ```

        Rules:
        - All edit blocks must appear within a markdown block.
        - Create targeted changes and avoid repeating large blocks of unchanged lines.
        - The first edit block must include the filename in the '--- search filename' line. Subsequent blocks for the same file can use '--- search' without the filename.
        - Search block: Include enough contiguous lines to uniquely identify the source context.
        - Replace block: Provide complete replacement content
        - Output only edit blocks. Do not add line numbers these are not part of the text of the file.

        ## ruff check output
        
        ruff check {fname.name}

        ```
        __output__
        ```
        """)
    msg = msg.replace("__output__", lint_output)
    msg += vfs.get_file(fname).as_markdown()

    # msg = msg.replace("__code__", fname.read_text(encoding="utf-8"))
    console.print(Markdown(msg))
    llm = LLM(OpenAIModel.GPT, dev_inst, True)
    history = MessageHistory()
    history.append(user_message(msg))
    response = llm.create(history)
    print_response(response)
    r, content, typ = extract_text_content(response)
    blocks = cu.extract_markdown_blocks(content)
    if not blocks and content.startswith("---"):
        blocks = [cu.CodeBlock(language="", lines=content.splitlines())]
    if blocks:
        msg2 = vfs.apply_edits(blocks)
        vfs.save_all()
        success, lint_output = cu.run_linter(fname)
        if success:
            msg2 += "\n\nsummarise the changes. do not repeat the code."
            history.append(user_message(msg2))
            print_message(msg2)
            response = llm.create(history)
            print_response(response)
        else:
            console.print(lint_output, style="red")
    else:
        console.print("no markdown blocks found", style="red")
    console.print(f"{llm.usage}")


def mypy_workflow(fname: Path):
    vfs.create_mapping(fname)
    success, output = cu.run_mypy(fname)
    # cu.print_block(s, True)
    if success:
        return

    dev_inst = dedent(f"""\
    The assistant is Marvin a senior Python developer. 
    You have access to 2 tools:
    - read_file: use the read_file tool to read a file.

    the current datetime is {datetime.now().isoformat()}
    """)
    msg = dedent(f"""\
        task: review mypy errors

        instructions:
        - for each error understand the root cause of the error
        - decide whether the error can be ignored
        - if it is possible to fix the error propose a code change.

        ## mypy output

        mypy {fname.name}

        ```
        __output__
        ```
        """)
    msg = msg.replace("__output__", output)
    msg += vfs.get_file(fname).as_markdown()
    console.print(Markdown(msg))

    llm = LLM(OpenAIModel.GPT, dev_inst, True)
    history = MessageHistory()
    history.append(user_message(msg))
    response = llm.create(history)
    print_response(response)


def test_code_edit():
    dev_inst = dedent(f"""\
    ## Role
    The assistant is a web design assistant. You work autonomously to complete the user requests using the tools provided. Limit your changes to only those explicitly needed for the task.
                
    ## Available Tools
    
    - read_text(filename, line_number, window_size, show_line_numbers)  
    Reads a window of *window_size* lines centered on *line_number* (1-based).
    Set *window_size* = 0 to read entire file
    Set *show_line_numbers* = True to prefix each line with its line number.

    ## How to format edits
    Output structured edits in this exact format:
                      
    ```
    --- search filename
    context line 1
    line 2
    --- replace
    context line 1
    inserted line
    --- end
    ```
    
    Rules:
    - All edit blocks must appear within a markdown block.
    - Create targeted changes and avoid repeating large blocks of unchanged lines.
    - The first edit block must include the filename in the '--- search filename' line. Subsequent blocks for the same file can use '--- search' without the filename.
    - Search block: Include enough contiguous lines to uniquely identify the source context.
    - Replace block: Provide complete replacement content

    the current datetime is {datetime.now().isoformat()}
    """)
    history = MessageHistory()
    dev_msg = developer_message(dev_inst)
    history.append(dev_msg)
    contents = dedent("""\
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width,initial-scale=1">
            <title></title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/water.css@2/out/water.css">
          </head>
          <body>
            <main>
            </main>
          </body>
        </html>
        """)

    prompt = dedent("""\
        ## task
        edit the file template.html to make the text colour #586e75 and background #fdf6e3
        add some page content about the planet Mercury.
        Identify all the places in the source file where changes are necessary. Remember to update the title.
        
        ## output format
        Output the changes as structured edit blocks. Avoid repeating unchanged lines
        """)
    prompt = dedent("""\
        ## task
        cli.py is a skeleton for a command line LLM chat application. The goal of this stage it to implement multi-line text entry from the user.
        complete the Python code cli.py in the file as per the comments
        
        ## output format
        Output the code changes to cli.py as structured edit blocks. Avoid repeating unchanged lines
        """)

    # vfs.create_unmapped("template.html", contents)
    # prompt += vfs.get_file("template.html").as_markdown()
    # vfs.create_mapping(Path("~/Documents/chats/temp.py").expanduser())
    vfs.create_mapping(Path("~/code/scripts/cli.py").expanduser())
    prompt += vfs.get_file("cli.py").as_markdown()

    msg = user_message(prompt)
    print_message(msg)
    history.append(msg)

    llm = LLM(OpenAIModel.GPT, "", True)
    llm.instructions = dev_inst
    response = llm.create(history)
    print_response(response)
    vfs.save_all()
    r, t, typ = extract_text_content(response)
    if blocks := cu.extract_markdown_blocks(t):
        s = vfs.apply_edits(blocks)
        cu.print_block(vfs.read_text("cli.py"), line_numbers=True)
        console.print(s, style="yellow")
        if s.startswith("SUCCESS"):
            vfs.save_all()
            msg = user_message(s + "\n\nSummarise the change for the user. Do not repeat the code.")
            print_message(msg)
            history.append(msg)
            response = llm.create(history)
            print_response(response)

    console.print(f"{llm.usage}")


def unittest_workflow(fname: Path):
    testname = fname.parent / f"tests/test_{fname.name}"
    vfs.create_mapping(fname)
    vfs.create_mapping(testname)
    vfs.create_mapping(Path("~/code/scripts/chatutils.py").expanduser())
    success, output = cu.run_python_unittest(fname)
    if success:
        return

    dev_inst = dedent(f"""\
## Role
The assistant is a python coding assistant. You work in python 3.12. You work autonomously to complete the user requests using the tools provided.

## Available Tools

- read_text(filename, line_number, window_size, show_line_numbers)  
Reads a window of *window_size* lines centered on *line_number* (1-based).
Set *window_size* = 0 to read entire file
Set *show_line_numbers* = True to prefix each line with its line number.

## Edit output format
Output structured edits in this exact format:
                
--- search filename
<contiguous source context to be replaced>
--- replace
<complete replacement content for the matched region>
--- end

Each block starts with the control line '--- search' and ends with '--- end'
                
Rules:
- The first block for a file must use `--- search filename`. Additional blocks for the same file may use `--- search` (no filename).
- The search block must contain exact, contiguous text copied from the file (no line numbers, no ellipses, no regex). Include enough unique context to avoid accidental matches; prefer 2–5 lines surrounding the change.
- The replace block is the full intended content for that region (use it to insert, modify, or delete):
- Insert: include the surrounding context in the search block and add the new lines at the appropriate position in the replace block.
- Delete: omit the lines to be removed from the replace block.
- Modify: present the edited lines in the replace block.
- Create targeted changes; do not repeat large unchanged sections.
- If no changes are needed, output a single line: NO CHANGES
                
the current datetime is {datetime.now().isoformat()}
        """)
    llm = LLM(OpenAIModel.GPT, "", True)
    llm.instructions = dev_inst
    history = MessageHistory()

    prompt = dedent("""\
        task: fix unittest errors. The code is toolutils.py and associated test test_toolutils.py

        instructions: 
        - understand the error message.
        - read the test code to understand the purpose of the test
        - read relevant source code
        - do not assume the test is correct
        - produce the required edit blocks in the specified format.
        - Avoid repeating unchanged sections of source code in the search and replace blocks.

        ## unittest output

        ```
        __output__
        ```

        """)

    prompt = prompt.replace("__output__", output)
    errors = tu.extract_error_locations(output)
    for e in errors:
        s = f"\n## {e.file_path}:{e.line_number}\n\n```python\n{''.join(e.source_code)}\n```\n\n"
        prompt += s
    prompt += cu.load_textfile(fname)
    msg = user_message(prompt)
    # cu.save_content(prompt)
    history.append(msg)
    print_message(msg)
    exit()
    response = llm.create(history)
    print_response(response)
    # vfs.save_all()
    r, t, typ = extract_text_content(response)
    if blocks := cu.extract_markdown_blocks(t):
        s = vfs.apply_edits(blocks)
        cu.print_block(vfs.read_text(fname.name, line_number=1, window_size=0), line_numbers=True)
        msg = user_message(f"{s}.\n\nSummarise the change. Do not repeat the source code")
        history.append(msg)
        print_message(msg)
        response = llm.create(history)
        print_response(response)
    else:
        console.print("No changes found", style="yellow")

    console.print(f"{llm.usage}", style="green")


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
    fname = Path("/code/scripts/chatutils.py")
    s = cu.run_linter(fname)
    if "error" in s.lower():
        prompt = dedent(f"""\
            ## task
            fix the issues identified by the Python linter in {fname.name}
            
            ## instructions
            - understand the lint output
            - make the changes to fix the lint issues. do not make any other changes.
            - summarise the diff for the user
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
        response_first = client.responses.parse(model=OpenAIModel.GPT_MINI, input=[user_message(InputText(text=question)).model_dump()], text_format=AnswerSheet)

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


def process_command(cmd: str, params: str) -> None:
    if cmd == "attach":
        p = Path(params).expanduser() if params.startswith("~") else Path(params)
        if vfs.create_mapping(p):
            console.print(f"attached file {p}", style="green")
        else:
            console.print(f"file {p} not found", style="red")
    elif cmd == "lint":
        p = Path(params + ".py").expanduser()
        if p.exists():
            lint_workflow(p)
        else:
            console.print(f"file {p} not found", style="red")
    else:
        console.print(f"unknown command: {cmd}", style="red")


def test_chat_loop():
    """a simple multi-turn conversation maintaining coversation state using response.id"""

    dev_inst = f"The assistant is Marvin an AI chatbot. The assistant uses precise language and avoids vague or generic statements. The current date is {datetime.now().isoformat()}"
    # dev_inst = dedent(f"""\
    #     The assistant is Marvin a helpful AI chatbot. Marvin is an expert python programmer.
    #     task: help the user write a design. the user will provides requirements and you should use those to formulate a design
    #     you will need to collaborate with the user to refine the design until the user is satisfied
    #     keep a track of all important decisions and the reasoning behind them
    #     The current date is {datetime.now().isoformat()}
    #     """)
    model = LLM(OpenAIModel.GPT_MINI, use_tools=True)
    history = MessageHistory()
    history.append(developer_message(dev_inst))
    history.print()
    inp = ""
    while True:
        inp = cu.input_multi_line()
        if inp == "x":
            break
        if inp[0] == "%":
            xs = inp.split(maxsplit=1)
            process_command(xs[0][1:], xs[1])
            continue

        if vfs:
            inp += "\n" + vfs.as_markdown()
        console.print(Markdown(f"user:\n\n{inp}\n"), style="green")
        history.append(user_message(inp))
        response = model.create(history)
        console.print(Markdown(f"assistant ({response.model}):\n\n{response.output_text}\n"), style="cyan")
        console.print(f"{model.usage}", style="yellow")

    history.print()
    # pprint(history)


def main():
    # simple_message()
    # test_search_example()
    # test_file_inputs()
    # test_function_calling()
    # test_function_calling_python()
    # test_function_calling_shell()
    # test_image_analysis()
    # test_solve_visual_maths_problem()
    # structured_output_message()
    git_workflow()
    # lint_workflow(Path("~/code/scripts/chatutils.py").expanduser())
    # mypy_workflow(Path("~/code/scripts/toolutils.py").expanduser())
    # unittest_workflow(Path("~/code/scripts/toolutils.py").expanduser())
    # test_code_edit()
    # test_apply_diff()
    # test_chat_loop()
    # failed,s = cu.run_python_unittest(Path("./toolutils.py"), "apply_diff_impl")


if __name__ == "__main__":
    main()
