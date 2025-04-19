import base64
import json
import re
from datetime import datetime
from functools import cache
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field, SerializeAsAny, field_validator
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint

import chatutils as cu

# ---
# examples using OpenAI responses API
# ---

console = Console()
client = OpenAI()

# types for message construction


class InputItem(BaseModel):
    type: str


class InputText(InputItem):
    type: Literal["input_text", "output_text"] = "input_text"
    text: str


class InputImage(InputItem):
    type: Literal["input_image"] = "input_image"
    image_url: str | Path

    @field_validator("image_url", mode="before")
    @classmethod
    def validate_image_url(cls, v):
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
    def validate_file_id(cls, v):
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
    role: Literal["user", "assistant"]
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


# helper functions for message construction


def user_message(*items: str | InputItem) -> Message:
    return Message(role="user", content=[InputText(text=item) if isinstance(item, str) else item for item in items])


def assistant_message(response) -> Message:
    r, s, t = extract_text_content(response)
    return Message(role=r, content=[InputText(type=t, text=s)])


def function_call(tool_call) -> FunctionCall:
    return FunctionCall(type=tool_call.type, id=tool_call.id, call_id=tool_call.call_id, name=tool_call.name, arguments=tool_call.arguments)


def function_call_output(tool_call, result) -> FunctionCallOutput:
    return FunctionCallOutput(call_id=tool_call.call_id, output=str(result))


def reasoning_item(item) -> ReasoningItem:
    return ReasoningItem(id=item.id, summary=item.summary, type=item.type)


class Answer(BaseModel):
    number: int = Field(description="question number")
    choice: str = Field(description="the single word choice")


# structured output models


class AnswerSheet(BaseModel):
    answers: list[Answer] = Field(description="the list of answers")

    def to_yaml(self) -> str:
        xs = (f"  - Q{x.number}: {x.choice}" for x in self.answers)
        return f"answers:\n{'\n'.join(xs)}"


class Marked(BaseModel):
    number: int = Field(description="question number")
    answer: str = Field(description="given choice")
    expected: str = Field(description="correct answer")
    feedback: str = Field(description="explanation for incorrect answers")
    is_correct: bool = Field(description="True if correct")


class MarkSheet(BaseModel):
    answers: list[Marked] = Field(description="list of marked questions")
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

    def create(self, history, msg):
        """create a message based on existing history. update history with assistant message"""
        if msg:
            history.append(msg)
        args = {"model": self.model, "input": [e.model_dump() for e in history]}
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
            console.print("returning function call results", style="yellow")
            args["input"] = [e.model_dump() for e in history]
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
    "description": "Use this tool to evaluate mathematical or Python expressions. Functions from the standard Python 3.12 library can be used.",
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


@cache
def fn_mapping() -> dict[str, dict[str, Any]]:
    """Returns a dictionary mapping function names to their definitions and a callable."""
    return {
        eval_fn["name"].lower(): {"defn": eval_fn, "fn": evaluate_expression},
        execute_script_fn["name"].lower(): {"defn": execute_script_fn, "fn": execute_script},
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
            r = f"ERROR: tool call failed with {type(e).__name__} - {str(e)}"
            console.print(r, style="red")
            return r

    r = f'ERROR: No tool named "{fn_name}" found'
    console.print(r, style="red")
    return r


def process_function_call(tool_call, history: list):
    """process the tool call and update the history"""
    console.print(f"{tool_call.type}: {tool_call.name} with {tool_call.arguments}", style="yellow")
    history.append(function_call(tool_call))
    # if we used parse then the arguments for the tool will have be extracted
    args = getattr(tool_call, "parsed_arguments", None)
    if not args:
        args = json.loads(tool_call.arguments)
    result = dispatch(tool_call.name, args)
    history.append(function_call_output(tool_call, result))


def process_function_calls(history, response):
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
                history.append(Message(role=output.role, content=[InputText(type="output_text", text=output_text)]))
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


def extract_text_content(response) -> tuple[str, str, str]:
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


def print_response(response) -> str:
    r, s, t = extract_text_content(response)
    console.print(Markdown(f"{r} ({response.model}):\n\n{s}\n"), style="cyan")
    return t


def execute_script(language: str, script_lines: list[str]) -> Any:
    code = cu.CodeBlock(language.lower(), script_lines)
    return cu.execute_script(code)


def evaluate_expression_impl(expression: str) -> Any:
    # Split into individual parts remove whitespace - only works if no indents
    parts = [e.strip() for e in re.split(r"[;\n]", expression)]
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

    # Evaluate and return the final expression
    return eval(last_part, namespace)


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
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=f"role: Marvin, a super intelligent AI chatbot. The current date is {datetime.now().isoformat()}",
        input="list the last 3 UK prime ministers give the month and year they became PM. Who is PM today?",
    )
    console.print(Markdown(response.output_text), style="cyan")


def test_file_inputs():
    """upload a pdf and ask questions about the content"""
    llm = LLM("gpt-4.1-mini", instructions="role: AI researcher")
    history = []

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

    msg = user_message("Summarise the thesis put forward. How relevant is it in 2024?")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)


def test_function_calling():
    llm = LLM("gpt-4.1-mini", f"Use the eval tool when necessary. the current datetime is {datetime.now().isoformat()}", True)
    # llm = LLM("gpt-4.1-mini", "use the eval tool as needed", True)
    history = []

    msg = user_message("a circle is inscribed in a square. the square has a side length of 3 m. what is the area inside the square but not in the circle.")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    msg = user_message("find the roots of $$2x^{2}-5x-6$$. express as decimals")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    msg = user_message("how many days since the first moon landing")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)


def test_function_calling_python():
    llm = LLM("o4-mini", f"Use the eval tool to evaluate expressions. Use the execute_script tool to run scripts on the local computer. the current datetime is {datetime.now().isoformat()}", True)
    history = []
    msg = user_message("is 30907 prime. if not what are its prime factors")
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    question = "Given 4 inequalities S > C, T > 2C, 2S > T + C, 2C > S where all are natural numbers less than 10, find possible values of C, S, T"
    msg = user_message(question)
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)


def test_function_calling_powershell():
    llm = LLM("gpt-4.1-mini", f"Use the eval tool to evaluate expressions. Use the execute_script tool to run scripts on the local computer. the current datetime is {datetime.now().isoformat()}", True)
    # llm = LLM("gpt-4.1-mini", "use the eval tool as needed", True)
    history = []

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
    """test image analysis of web image and local uploaded image"""
    msg = user_message(InputText(text="what is in the foreground and background of this image. describe the outfit"), InputImage(image_url="https://i.dailymail.co.uk/1s/2024/06/08/22/85884439-0-image-a-144_1717882540270.jpg"))
    response = client.responses.create(model="gpt-4.1-mini", input=[msg.model_dump()])
    print_response(response)
    pprint(response.usage)

    msg = user_message(InputText(text="extract the paragraph of text and put inside a markdown block enclosed with <text> tags. describe the balances scales in this drawing"), InputImage(image_url=Path.home() / "Downloads" / "sum1.png"))
    response = client.responses.create(model="gpt-4.1", input=[msg.model_dump()])
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
    """Answers question returning json output then reformat to yaml and get it marked returning json"""

    def process_response_output(response_outputs):
        ans = ""
        # ignore output of type ResponseReasoningItem which are output by reasoning models
        for output in (x for x in response_outputs if x.type == "message"):
            for item in output.content:
                s = f"{output.role}:\n\n{item.text}" if item.type == "output_text" else f"{output.role}:\n\n{item.type}"
                console.print(Markdown(s), style="cyan")
                # parsed only exists if the return type is ParsedResponseOutputText
                if parsed := getattr(item, "parsed", None):
                    ans = f"```yaml\n{parsed.to_yaml()}\n```"
                    console.print(Markdown(ans), style="cyan")
        return ans

    def ask_question_and_mark(question: str, answer_template: str) -> None:
        console.print(Markdown(question), style="green")

        # First parse: retrieve response from model "o4-mini" with low reasoning effort
        #        response_first = client.responses.parse(model="gpt-4.1-nano", reasoning={"effort": "low"}, input=[user_message(InputText(text=question)).model_dump()], text_format=AnswerSheet)
        response_first = client.responses.parse(model="gpt-4.1-mini", input=[user_message(InputText(text=question)).model_dump()], text_format=AnswerSheet)

        # Convert returned json to YAML format using process_response_output
        yaml_ans = process_response_output(response_first.output)

        # Replace the placeholder "{input}" in the answer template with the YAML output
        inp = answer_template.replace("{input}", yaml_ans)
        console.print(Markdown(inp), style="green")

        # Second parse: pass the updated answer input to "gpt-4.1-mini" model with MarkSheet format
        response_final = client.responses.parse(model="gpt-4.1-nano", input=[user_message(InputText(text=inp)).model_dump()], text_format=MarkSheet)

        # Process and print the final output
        process_response_output(response_final.output)

    root = Path.home() / "Documents" / "chats"
    # extract the questions from original and insert into new template to allow different instructions
    q5 = extract_text_block(root / "q5.md")
    q5 = questions_q5.replace("{input}", q5)
    q5_ans = extract_text_block(root / "q5-ans.md")
    q5_ans = answers_q5.replace("{answers}", q5_ans)
    ask_question_and_mark(q5, q5_ans)

    # Process q6
    q6 = extract_text_block(root / "q6.md")
    q6 = questions_q6.replace("{input}", q6)
    q6_ans = extract_text_block(root / "q6-ans.md")
    q6_ans = answers_q6.replace("{answers}", q6_ans)
    ask_question_and_mark(q6, q6_ans)


def main():
    # simple_message()
    # test_file_inputs()
    # test_image_analysis()
    # test_solve_visual_maths_problem()
    structured_output_message()
    # test_function_calling()
    # test_function_calling_powershell()
    # test_function_calling_python()
    # test_pdf_analysis()


if __name__ == "__main__":
    main()
