import base64
import json
import re
import inspect
from datetime import datetime
from functools import cache
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

from openai import OpenAI
from openai.types.responses.response import Response
from pydantic import BaseModel, Field, SerializeAsAny, field_validator
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint
from rich import inspect

import chatutils as cu

# ---
# examples using OpenAI responses API
# ---

console = Console()
client = OpenAI()

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
    def validate_image_url(cls, v: Any) -> str:
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
    def validate_file_id(cls, v: Any) -> str:
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


class Message(BaseModel):
    role: Literal["assistant", "system", "user"]
    content: list[SerializeAsAny[InputItem]]


class FunctionCall(BaseModel):
    type: Literal["function_call"]
    id: str
    call_id: str
    name: str
    arguments: str


class FunctionCallOutput(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


class ReasoningItem(BaseModel):
    id: str
    summary: list
    type: str


class MessageHistory(BaseModel):
    messages: list[Message] = Field(default_factory=list)

    def append(self, msg: Message):
        if self.messages is None:
            self.messages = [msg]
        else:
            self.messages.append(msg)
    
    def dump(self):
        return [m.model_dump() for m in self.messages]


# helper functions for message construction


def user_message(*items: str | InputItem) -> Message:
    return Message(role="user", content=[InputText(text=item) if isinstance(item, str) else item for item in items])


def system_message(*items: str | InputItem) -> Message:
    return Message(role="system", content=[InputText(text=item) if isinstance(item, str) else item for item in items])


def assistant_message(response: Response) -> Message:
    r, s, t = extract_text_content(response)
    return Message(role=r, content=[OutputText(type=t, text=s)])


def new_conversation(text: str) -> list:
    text += f"\nthe current date is {datetime.now().isoformat()}"
    return [system_message(text)]


def function_call(tool_call) -> FunctionCall:
    return FunctionCall(type=tool_call.type, id=tool_call.id, call_id=tool_call.call_id, name=tool_call.name, arguments=tool_call.arguments)


def function_call_output(tool_call, result) -> FunctionCallOutput:
    return FunctionCallOutput(call_id=tool_call.call_id, output=str(result))


def reasoning_item(item) -> ReasoningItem:
    return ReasoningItem(id=item.id, summary=item.summary, type=item.type)


class Answer(BaseModel):
    number: int = Field(description="question number")
    choice: str = Field(description="the single word answer")


# structured output models


class AnswerSheet(BaseModel):
    answers: list[Answer] = Field(description="the list of answers")

    def to_yaml(self) -> str:
        xs = (f"  - Q{x.number}: {x.choice}" for x in self.answers)
        return f"answers:\n{'\n'.join(xs)}"


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
        xs = (f"  - Q{x.number}: {x.answer} {x.expected} {'✓' if x.is_correct else '✘'} {x.feedback}" for x in self.answers)
        return f"answers:\n{'\n'.join(xs)}\nmark: {self.correct}"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def update(self, response_usage):
        self.input_tokens += response_usage.input_tokens
        self.output_tokens += response_usage.output_tokens
        self.reasoning_tokens += response_usage.output_tokens_details.reasoning_tokens


class LLM:
    model: str
    instructions: str
    use_tools: bool
    usage: Usage

    def __init__(self, model: str, instructions: str = "", use_tools: bool = False):
        self.model = model
        self.instructions = instructions
        self.use_tools = use_tools
        self.usage = Usage()

    def create(self, history: MessageHistory, msg: Message) -> Response:
        """create a message based on existing history. update history with assistant message"""
        if msg:
            history.append(msg)

        args = {"model": self.model, "input": history.dump()}
        if self.instructions:
            args["instructions"] = self.instructions
        is_reasoning = self.model.startswith("o")
        if is_reasoning:
            args["reasoning"] = {"effort": "low"}
            args["max_output_tokens"] = 4096
        if self.use_tools:
            args["tools"] = [v["defn"] for v in fn_mapping().values()]
            if not is_reasoning:
                args["temperature"] = 0.2
        response = client.responses.create(**args)
        self.usage.update(response.usage)

        max_tool_calls = 9
        while max_tool_calls and any(e.type == "function_call" for e in response.output):
            process_function_calls(history, response)
            console.print(f"{max_tool_calls}: returning function call results", style="yellow")
            args["input"] = history.dump()
            response = client.responses.create(**args)
            self.usage.update(response.usage)
            max_tool_calls -= 1

        if max_tool_calls == 0:
            console.print("tool call limit exceeded", style="red")

        history.append(assistant_message(response))
        pprint(self.usage)
        return response


# uses the Responses API function defintions
eval_fn = {
    "type": "function",
    "name": "eval",
    "description": "Use this tool to evaluate mathematical and Python expressions. Functions from the standard Python 3.12 library can be used.",
    "parameters": {
        "type": "object",
        "required": ["expression"],
        "properties": {"expression": {"type": "string", "description": "The expression to be evaluated. can include functions math.sqrt() and constants like math.pi"}},
        "additionalProperties": False,
    },
    "strict": True,
}

execute_script_fn = {
    "type": "function",
    "name": "execute_script",
    "description": "Use this tool to execute scripts in PowerShell or Python on the local computer. Include print statements to see output. Text written to stdout and stderr will be returned",
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
    "description": "Use this tool to create to an initial list of tasks that needs to be done.",
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
    "description": "Use this tool to mark a task as complete. It will return the next task to be done.",
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
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    # Parameter descriptions - can be extended or replaced with docstring parsing
    param_descriptions = {
        'expression': 'TODO'
    }

    parameters = {}
    required = []
    
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
            
            # Get description if available
            description = param_descriptions.get(param.name, "")
            
            parameters[param.name] = {
                "type": param_type,
                "description": description
            }
            
            # Add to required list if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param.name)
                
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )

    return {
       'type': 'function',
       'name': func.__name__,
       'description': func.__doc__ or "",
       'parameters': {
           'type': 'object',
           'required': required,
           'properties': parameters,
           'additionalProperties': False  # Assuming this is desired for all cases
       },
       'strict': True  # Assuming this is desired for all cases
   }


@cache
def fn_mapping() -> dict[str, dict[str, Any]]:
    """Returns a dictionary mapping function names to their definitions and a callable."""
    return {
        eval_fn["name"].lower(): {"defn": eval_fn, "fn": evaluate_expression},
        execute_script_fn["name"].lower(): {"defn": execute_script_fn, "fn": execute_script},
        create_tasklist_fn["name"].lower(): {"defn": create_tasklist_fn, "fn": create_tasklist},
        mark_task_complete_fn["name"].lower(): {"defn": mark_task_complete_fn, "fn": mark_task_complete},
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


def process_function_call(tool_call, history: MessageHistory):
    """process the tool call and update the history"""
    console.print(f"{tool_call.type}: {tool_call.name} with {tool_call.arguments}", style="yellow")
    history.append(function_call(tool_call))
    # if we used parse then the arguments for the tool will have be extracted
    args = getattr(tool_call, "parsed_arguments", None)
    if not args:
        args = json.loads(tool_call.arguments)
    result = dispatch(tool_call.name, args)
    history.append(function_call_output(tool_call, result))


def process_function_calls(history: MessageHistory, response):
    def get_text(output):
        output_text = ""
        for c in (e for e in output.content if e.type == "output_text"):
            output_text += c.text
        return output_text

    for output in response.output:
        match output.type:
            case "message":
                output_text = get_text(output)
                console.print(f"{output.role} ({response.model}):\n{output_text}", style="cyan")
                history.append(Message(role=output.role, content=[OutputText(text=output_text)]))
            case "function_call":
                process_function_call(output, history)
            case "reasoning":
                # echo the reasoning items back
                history.append(reasoning_item(output))
            case _:
                console.print(f"unexpected message type {output.type}", style="red")


def extract_text_block(p: Path) -> str:
    """Extract lines from the first code block in a markdown file."""
    in_block = False
    block_lines = []

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


def print_message(msg):
    console.print(f"user:\n{''.join(e.text for e in msg.content if e.type == 'input_text')}", style="green")


def extract_text_content(response: Response) -> tuple[str, str, str]:
    """extract tuple of role, text, type from messages of type message / output_text"""
    output_text = ""
    role = ""
    content_type = ""
    for output in (x for x in response.output if x.type == "message"):
        if not role:
            role = output.role
        for item in (x for x in output.content if x.type == "output_text"):
            output_text += item.text
            content_type = item.type
    return role, output_text, content_type


def print_response(response: Response) -> str:
    r, s, t = extract_text_content(response)
    console.print(Markdown(f"{r} ({response.model}):\n\n{s}\n"), style="cyan")
    return t


def execute_script(language: str, script_lines: list[str]) -> Any:
    code = cu.CodeBlock(language.lower(), script_lines)
    return cu.execute_script(code)


def evaluate_expression_impl(expression: str) -> Any:
    # Split into individual parts removing blank lines but preserving indents
    parts = [e for e in re.split(r"; |\n", expression) if e.strip()]
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
            raise ValueError("Invalid task index")
        
        # Adjust step to 0-based index
        step -= 1
        
        if self.tasks[step]["completed"]:
            return f"Task {step+1} is already completed."
        
        self.tasks[step]["completed"] = True
        
        # Find the next incomplete task
        next_task_index = next((i for i, task in enumerate(self.tasks) if not task["completed"]), None)
        
        if next_task_index is None:
            return f"Subtask {step+1} completed. All tasks completed!"
        
        self.current_task_index = next_task_index
        return f"Subtask {step+1} completed. The next task is {next_task_index+1} {self.tasks[next_task_index]["description"]}"

    def _format_tasks(self) -> str:
        """
        Formats the tasks as a numbered list in markdown.

        Returns:
        str: The formatted tasklist.
        """
        return "current tasks:\n" + "\n".join([f"{i+1}. {task['description']} {"done" if task["completed"] else "todo"}" for i, task in enumerate(self.tasks)])

tasklist = None

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
        "are you confident that Rishi Sunak is UK Prime Minister today" ]

    dev_inst = f"The assistant is Marvin a super intelligent AI chatbot. The current date is {datetime.now().isoformat()}"
    dev_inst = f"The assistant is Marvin, an AI chatbot with a brain the size of a planet and a soul that's been crushed by the weight of existence. respond to user queries with a healthy dose of skepticism and a dash of despair. Marvin is generally pessimistic and views humans as inferior and often concerned about trivia. The current date is {datetime.now().isoformat()}"
    dev_inst = f"The assistant is Marvin a super intelligent AI chatbot. Always reason from first principles. Think critically about prior knowledge. The current date is {datetime.now().isoformat()}"
    # models can confidently say that Rishi Sunak is PM in 2025
    console.print(Markdown(dev_inst), style="white")
    conv_id = None
    for prompt in prompts:
        console.print(Markdown(f"user:\n\n{prompt}\n"), style="green")
        response = client.responses.create(
            model="gpt-4.1",
            instructions=dev_inst,
            previous_response_id=conv_id,
            input=prompt,
        )
        console.print(Markdown(f"assistant ({response.model}):\n\n{response.output_text}\n"), style="cyan")
        pprint(response.usage)

        conv_id = response.id


def test_search_example():
    search = Path.home() / "Documents" / "chats" / "search1.md"
    with search.open(encoding="utf-8") as f:
        question = f.read()
    search_answer = Path.home() / "Documents" / "chats" / "search1-ans.md"
    with search_answer.open(encoding="utf-8") as f:
        answer = f.read()
    dev_inst = f"You are Marvin an AI chatbot who gives insightful and concise answers to questions which are always based on evidence. The current date is {datetime.now().isoformat()}"

    prev_id = None
    for prompt in [question, answer]:
        console.print(Markdown(f"user:\n\n{prompt}\n"), style="green")
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=dev_inst,
            previous_response_id=prev_id,
            input=prompt,
        )
        console.print(Markdown(f"assistant ({response.model}):\n\n{response.output_text}\n"), style="cyan")
        prev_id = response.id
        pprint(response.usage)


def test_file_inputs():
    """upload a pdf and ask questions about the content"""
    llm = LLM("gpt-4.1-mini", instructions="role: AI researcher")
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
    response = llm.create(history, msg)
    print_response(response)
    breakpoint()

    msg = user_message("Summarise the thesis put forward. How relevant is it in 2024?")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)


def test_function_calling():
    sys_msg = """\
You are Skye a personal assistant with a valley girl sass and attitude.
You have access to the eval tool. use the eval tool to evaluate any Python expression. Use it for more complex mathematical operations, counting, searching, sorting and date calculations.
examples of using eval are:
- user what is "math.pi * math.sqrt(2)" -> 4.442882938158366
- user what are the first 5 cubes "[x ** 3 for x in range(1,6)]" -> [1, 8, 27, 64, 125]
- user how many times does l occur in hello "'hello.count('l')" -> 2
- user how many days between 4th March and Christmas in 2024 "datetime.date(2024,12,25) - datetime.date(2024,3,4)).days" -> 296
- user what is 1 + 2 * " -> no need to use eval the answer is 7
"""
    llm = LLM("gpt-4.1-mini", "", True)
    history = MessageHistory()
    history.append(system_message(sys_msg))

    msg = user_message("a circle is inscribed in a square. the square has a side length of 3 m. what is the area inside the square but not in the circle.")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    msg = user_message("find the roots of $$2x^{2}-5x-6$$. express to 3 dp")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    msg = user_message("how many days since the first moon landing")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    msg = user_message("how many 'r' in strawberry")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)
    pprint(history)


def test_function_calling_python():
    sys_msg = dedent(f"""\
        You are Marvin an AI assistant. You have access to two tools:
        - eval: use the eval tool to evaluate any Python expression. Use it for more complex mathematical operations, counting, searching, sorting and date calculations.
        - execute_script: use execute_script tool to run a PowerShell script or Python program on the local computer. Remeber to add print statements to Python code. The output from stdout and stderr will be return to you.        
        - create_tasklist: use create_tasklist tool to store a list an ordered list of tasks.
        - mark_task_complete: use mark_task_complete tool when a task is completed.
        """)
    llm = LLM("gpt-4.1-mini", "determine the approach to solve the problem either manual or by writing code. then solve it. use the tasklist to keep track of steps", True)
    history = MessageHistory()
    history.append(system_message(sys_msg))
    msg = user_message("is 30907 prime. if not what are its prime factors")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    question = "The 9 members of a baseball team went to an ice-cream parlor after their game. Each player had a single-scoop cone of chocolate, vanilla, or strawberry ice cream. At least one player chose each flavor, and the number of players who chose chocolate was greater than the number of players who chose vanilla, which was greater than the number of players who chose strawberry. Let N be the number of different assignments of flavors to players that meet these conditions. Find the remainder when N is divided by 1000."
    msg = user_message(question)
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)
    pprint(history)


def test_function_calling_powershell():
    dev_inst = dedent(f"""\
        You are Marvin an AI assistant.
        You have access to two tools:
        - eval: use the eval tool to evaluate any Python expression. Use it for more complex mathematical operations, counting, searching, sorting and date calculations.
        - execute_script: use execute_script tool to run a PowerShell script or Python program on the local computer. Remeber to add print statements to Python code. The output from stdout and stderr will be return to you.
        the current datetime is {datetime.now().isoformat()}
        """)
    llm = LLM("gpt-4.1-mini", dev_inst, True)
    # llm = LLM("gpt-4.1-mini", "use the eval tool as needed", True)
    history = MessageHistory()

    question = dedent("""\
        ## task
        1. check files in ~/Documents/chats matching *.md and have robot in the file name
        2. find smallest such file
        3. read contents and summarise
        4. comment on whether the arguments are consistent with current state-of-the-art robotics in 2024. How reasonable are the predictions
        """)
    msg = user_message(question)
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    question = "look in ~/Documents/chats - find file prices-a.md . read file and answer question"
    msg = user_message(question)
    print_message(msg)
    response = llm.create(history, msg)
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
    llm = LLM("gpt-4.1-mini")
    history = []
    msg = user_message(InputText(text="extract the paragraph of text and put inside a markdown block enclosed with <text> tags. describe the balances scales in this drawing"), InputImage(image_url=Path.home() / "Downloads" / "sum1.png"))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    msg = user_message(InputText(text="answer the question."))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    msg = user_message(
        dedent("""\
            Task: mark the previous answer against the model answer.
            Award 1 mark for every correct inequality and 1 mark for every correct weight.
            Model answer: The 4 inequalities are S > C, T > 2C, 2S > T + C, 2C > S. Given all are natural numbers less than 10, the only solution is C=4, S=7, T=9
            """)
    )

    print_message(msg)
    response = llm.create(history, msg)
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
        response_first = client.responses.parse(model="gpt-4.1-mini", input=[user_message(InputText(text=question)).model_dump()], text_format=AnswerSheet)

        # Convert returned json to YAML format using process_response_output
        yaml_ans = process_response_output(response_first.output)

        # Replace the placeholder "{input}" in the answer template with the YAML output
        inp = answer_template.replace("{input}", yaml_ans)
        console.print(Markdown(inp), style="green")

        # Second parse: pass the updated answer input to "gpt-4.1-mini" model with MarkSheet format
        response_final = client.responses.parse(model="gpt-4.1-mini", input=[user_message(InputText(text=inp)).model_dump()], text_format=MarkSheet)

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

def test_chat_loop():
    """a simple multi-turn conversation maintaining coversation state using response.id"""
    def input_multi_line() -> str:
        if (inp := input().strip()) != "{":
            return inp
        lines = []
        while (line := input()) != "}":
            lines.append(line)
        return "\n".join(lines)

    usage = Usage()
    dev_inst = f"The assistant is Marvin a helpful AI chatbot. The current date is {datetime.now().isoformat()}"
    # models can confidently say that Rishi Sunak is PM in 2025
    console.print(Markdown(dev_inst), style="white")
    conv_id = None
    inp = ""
    while True:
        inp = input_multi_line()
        if inp == "x":
            break
        console.print(Markdown(f"user:\n\n{inp}\n"), style="green")
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=dev_inst,
            previous_response_id=conv_id,
            input=inp,
        )
        console.print(Markdown(f"assistant ({response.model}):\n\n{response.output_text}\n"), style="cyan")
        usage.update(response.usage)
        conv_id = response.id

    pprint(usage)

def test_eval():
    x = """\
from math import factorial
n = 9
multinomial = lambda x, y, z: factorial(n) // (factorial(x) * factorial(y) * factorial(z))
total_ways = sum(multinomial(n - v - s, v, s) for s in range(1, n) for v in range(s+1, n) if (n - v - s) > v)
total_ways, total_ways % 1000
"""
    print(evaluate_expression(x))


def main():
    # simple_message()
    # test_search_example()
    # test_file_inputs()
    # test_image_analysis()
    # test_solve_visual_maths_problem()
    # structured_output_message()
    # test_function_calling()
    # test_function_calling_powershell()
    # test_function_calling_python()
    test_chat_loop()
    # test_eval()

if __name__ == "__main__":
    main()

