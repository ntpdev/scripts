import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import chatutils as cu
from openai import OpenAI
from pydantic import BaseModel, Field, SerializeAsAny, field_validator
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint

# examples using OpenAI responses API
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
    def _encode_if_path(cls, v):
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
    def _upload_if_path(cls, v):
        if isinstance(v, Path):
            xs = client.files.list(purpose="user_data")
            file = next((e for e in xs if e.filename == v.name), None)
            if not file:
                console.print(f"uploading file {v}", style="yellow")
                file = client.files.create(file=open(v, "rb"), purpose="user_data")
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

def user_message(*items: InputItem) -> Message:
    return Message(role="user", content=list(items))


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

# structure output models

class AnswerSheet(BaseModel):
    answers: list[Answer] = Field(description="the list of answers")

    def to_yaml(self) -> str:
        xs = [f"  - Q{x.number}: {x.choice}" for x in self.answers]
        return "answers:\n" + "\n".join(xs)


class Marked(BaseModel):
    number: int = Field(description="question number")
    answer: str = Field(description="given choice")
    expected: str = Field(description="correct answer")
    feedback: str = Field(description="explanation for incorrect answers")
    is_correct: str = Field(description="yes or no")


class MarkSheet(BaseModel):
    answers: list[Marked] = Field(description="list of marked questions")
    correct: int = Field(description="count of correct answers")

    def to_yaml(self) -> str:
        xs = [f"  - Q{x.number}: {x.answer} {x.expected} {'✓' if x.is_correct == 'yes' else '✘'} {x.feedback}" for x in self.answers]
        return "answers:\n" + "\n".join(xs) + f"\nmark {self.correct} / {len(self.answers)}"


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
    tools: list
    use_tools: bool
    usage: Usage

    def __init__(self, model, instructions=None, tools=None, use_tools=False):
        self.model = model
        self.instructions = instructions
        self.tools = tools
        self.use_tools = use_tools
        self.usage = Usage()

    def process_function_call(self, tool_call, history: list):
        """process the tool call and update the history"""
        console.print(f"{tool_call.type}: {tool_call.name} with {tool_call.arguments}", style="yellow")
        try:
            if tool_call.name == "eval":
                # echo the original function call back
                history.append(function_call(tool_call))
                # if we used parse then the arguments for the tool will have be extracted
                args = getattr(tool_call, "parsed_arguments", None)
                if not args:
                    args = json.loads(tool_call.arguments)
                result = evaluate_expression(**args)
                history.append(function_call_output(tool_call, result))
            elif tool_call.name == "execute_script":
                # echo the original function call back
                history.append(function_call(tool_call))
                # if we used parse then the arguments for the tool will have be extracted
                args = getattr(tool_call, "parsed_arguments", None)
                if not args:
                    args = json.loads(tool_call.arguments)
                result = execute_script(**args)
                if not result:
                    result = "ERROR: the script ran successfully but no output shown. add print statements to see values"
                history.append(function_call_output(tool_call, result))

        except Exception as e:
            result = "ERROR: " + str(e)
            console.print(result, style="red")
            history.append(function_call_output(tool_call, result))

    def process_function_calls(self, history, response):
        for output in response.output:
            if output.type == "message":
                output_text = get_text(output)
                console.print(f"{output.role} ({response.model}):\n{output_text}", style="cyan")
                history.append(Message(role=output.role, content=[InputText(type="output_text", text=output_text)]))
            elif output.type == "function_call":
                self.process_function_call(output, history)
            elif output.type == "reasoning":
                # echo the reasoning items back
                history.append(reasoning_item(output))
            else:
                console.print(f"unexpected message type {output.type}", style="red")

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
        if self.tools and self.use_tools:
            args["tools"] = self.tools
            if not is_reasoning:
                args["temperature"] = 0.2
        response = client.responses.create(**args)
        self.usage.update(response.usage)

        max_tool_calls = 9
        while max_tool_calls and any(e.type == "function_call" for e in response.output):
            self.process_function_calls(history, response)
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


eval_fn = {
    "type": "function",
    "name": "eval",
    "description": "Use this tool to evaluate mathematical or Python expressiona. The Python version 3.12",
    "parameters": {"type": "object", "required": ["expression"], "properties": {"expression": {"type": "string", "description": "The expression to be evaluated"}}, "additionalProperties": False},
    "strict": True,
}

execute_script_fn = {
    "type": "function",
    "name": "execute_script",
    "description": "Use this tool to execute PowerShell commands on the local computer. PowerShell version 7.4",
    "parameters": {"type": "object", "required": ["language", "script_lines"], "properties": {"language":{"type": "string", "enum": ["PowerShell", "Python"] , "description": "the name of the scripting language either PowerShell or Python"} , "script_lines": {"type": "array", "description": "The list of lines", "items":{ "type": "string", "description":"a line"} }}, "additionalProperties": False},
    "strict": True,
}


questions_q5 = """
## question
for each question choose a single word from the Choices list that has the most similar meaning to words in lists A and B.

```yaml
Q1:
  - A:
    - point
    - direct
  - B:
    - purpose
    - intention
  - Choices:
    - goal
    - aim
    - motive
    - guide
Q2:
  - A:
    - trench
    - drain
  - B:
    - abandon
    - dump
  - Choices:
    - drop
    - gutter
    - ditch
    - leave
Q3:
  - A:
    - talent
    - ability
  - B:
    - present
    - offering
  - Choices:
    - gift
    - bonus
    - skill
    - flair
Q4:
  - A:
    - path
    - route
  - B:
    - hunt
    - pursue
  - Choices:
    - way
    - chase
    - passage
    - track
Q5:
  - A:
    - law
    - code
  - B:
    - lead
    - reign
  - Choices:
    - rule
    - govern
    - order
    - instruct
Q6:
  - A:
    - assess
    - grade
  - B:
    - scratch
    - dent
  - Choices:
    - judge
    - mark
    - rate
    - cut
Q7:
  - A:
    - people
    - tribe
  - B:
    - run
    - sprint
  - Choices:
    - dash
    - nation
    - race
    - type
Q8:
  - A:
    - late
    - overdue
  - B:
    - after
    - following
  - Choices:
    - delayed
    - next
    - detained
    - behind
Q9:
  - A:
    - stone
    - boulder
  - B:
    - swing
    - sway
  - Choices:
    - roll
    - rock
    - tilt
    - cobble
Q10:
  - A:
    - bolt
    - fasten
  - B:
    - ringlet
    - curl
  - Choices:
    - hair
    - seal
    - plait
    - lock
Q11:
  - A:
    - find
    - discover
  - B:
    - stain
    - blemish
  - Choices:
    - freckle
    - smudge
    - spot
    - see
```
"""

answers_q5 = """
## task
check the student answers against this list of correct answers. mark each answer as either yes or no. For wrong answers provide feedback explaining why the correct word is a better fit. no comment is required for correct answers. give the final score.

## correct answers
```yaml
answers:
  - Q1: aim
  - Q2: ditch
  - Q3: gift
  - Q4: track
  - Q5: rule
  - Q6: mark
  - Q7: race
  - Q8: behind
  - Q9: rock
  - Q10: lock
  - Q11: spot
```

## student answers
{input}
"""

questions_q6 = """
## task
for each question choose the word from the choices list that means the same or nearly the same a the first word

```yaml
Q1:
  word: brave
  choices:
    - noble
    - fearless
    - capable
    - tough
Q2:
  word: guess
  choices:
    - consider
    - estimate
    - belief
    - idea
Q3:
  word: worth
  choices:
    - payment
    - value
    - expensive
    - reward
Q4:
  word: study
  choices:
    - teach
    - student
    - learn
    - education
Q5:
  word: assist
  choices:
    - treat
    - remedy
    - cure
    - aid
Q6:
  word: glad
  choices:
    - comforted
    - pleased
    - comical
    - witty
Q7:
  word: shovel
  choices:
    - excavate
    - sift
    - tunnel
    - scoop
Q8:
  word: ring
  choices:
    - around
    - shape
    - band
    - tunnel
Q9:
  word: bendy
  choices:
    - broken
    - flexible
    - snap
    - springy
Q10:
  word: tight
  choices:
    - taut
    - solid
    - immovable
    - tense
Q11:
  word: scare
  choices:
    - afraid
    - shock
    - frightful
    - fluster
Q12:
  word: shrink
  choices:
    - slight
    - miniature
    - wane
    - simplify
Q13:
  word: soak
  choices:
    - drench
    - damp
    - drip
    - clean
Q14:
  word: blunt
  choices:
    - curt
    - silent
    - secretive
    - defensive
```
"""

answers_q6 = """
## task
check the previous answers against this list of correct answers. mark the original answers as either yes or no. No comment is required for correct answers. For wrong answers provide feedback on why it is not the best choice. give the final score out of 14.

## correct answers
```yaml
answers:
  - Q1: fearless
  - Q2: estimate
  - Q3: value
  - Q4: learn
  - Q5: aid
  - Q6: pleased
  - Q7: scoop
  - Q8: band
  - Q9: flexible
  - Q10: taut
  - Q11: shock
  - Q12: wane
  - Q13: drench
  - Q14: curt
```

## student answers
{input}
"""


def extract_text_content(response) -> str:
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


def print_message(msg):
    console.print(f"user:\n{''.join(e.text for e in msg.content if e.type == 'input_text')}", style="green")


def print_response(response):
    # pprint(response)
    r, s, t = extract_text_content(response)
    console.print(Markdown(f"{r} ({response.model}):\n\n{s}\n"), style="cyan")


def simple_message():
    response = client.responses.create(
        model="gpt-4.1",
        instructions=f"role: Marvin, a super intelligent AI chatbot. The current date is {datetime.now().isoformat()}",
        input="list the last 3 UK prime ministers give the month and year they became PM. Who is PM today?",
    )
    console.print(Markdown(response.output_text), style="cyan")


def test_file_inputs():
    """upload a pdf and ask questions about the content"""
    llm = LLM("gpt-4.1-mini", instructions="role: AI researcher")
    history = []

    questions = """answer the following questions based on information in the file.
1. what games are discussed
2. what competition is mentioned
3. is a chess player mentioned
4. what was the name of the computer chess program
"""
    pdf = Path.home() / "Downloads" / "bitter_lesson.pdf"
    msg = user_message(InputText(text=questions), InputFile(file_id=pdf))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    msg = user_message(InputText(text="Summarise the thesis put forward. How relevant is it in 2024?"))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

def execute_script(language: str, script_lines: list[str]) -> Any:
    code = cu.CodeBlock(language.lower(), script_lines)
    return cu.execute_script(code)

def evaluate_expression_impl(expression: str) -> Any | None:
    # Split into individual parts won't remove whitespace in order to support multiline scripts
    parts = re.split(r"[;\n]", expression)
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


def evaluate_expression(expression: str) -> any:
    result = ""
    if expression:
        try:
            console.print("eval: " + expression, style="yellow")
            result = evaluate_expression_impl(expression)
            console.print("result: " + str(result), style="yellow")
        except Exception as e:
            result = "ERROR: " + str(e)
            console.print(result, style="red")
    else:
        result = "ERROR: no expression found"
        console.print(result, style="red")
    return result


def get_text(output):
    output_text = ""
    for c in (e for e in output.content if e.type == "output_text"):
        output_text += c.text
    return output_text


def test_function_calling():
    llm = LLM("gpt-4.1-mini", f"Use the eval tool when necessary. the current datetime is {datetime.now().isoformat()}", [eval_fn], True)
    # llm = LLM("gpt-4.1-mini", "use the eval tool as needed", [eval_fn], True)
    history = []

    msg = user_message(InputText(text="a circle is inscribed in a square. the square has a side length of 3 m. what is the area inside the square but not in the circle."))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    msg = user_message(InputText(text="find the roots of $$2x^{2}-5x-6$$. express as decimals"))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    msg = user_message(InputText(text="how many days since the first moon landing"))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

def test_function_calling_python():
    llm = LLM("gpt-4.1",
               f"Use the eval tool to evaluate expressions. Use the execute_script tool to run scripts on the local computer. the current datetime is {datetime.now().isoformat()}",
                 [eval_fn, execute_script_fn], True)
    history = []
    msg = user_message(InputText(text="is 30907 prime. if not what are its prime factors"))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    question = "Given 4 inequalities S > C, T > 2C, 2S > T + C, 2C > S where all are natural numbers less than 10, find possible values of C, S, T"
    msg = user_message(InputText(text=question))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)


def test_function_calling_powershell():
    llm = LLM("o4-mini",
               f"Use the eval tool to evaluate expressions. Use the execute_script tool to run scripts on the local computer. the current datetime is {datetime.now().isoformat()}",
                 [eval_fn, execute_script_fn], True)
    # llm = LLM("gpt-4.1-mini", "use the eval tool as needed", [eval_fn], True)
    history = []

    question = """
## task
1. check files in ~/Documents/chats matching *.md and have robot in the file name
2. find smallest such file
3. read contents and summarise
4. comment on whether the arguments are consistent with current state-of-the-art robotics in 2024. How reasonable are the predictions"
"""
    msg = user_message(InputText(text=question))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    question = "look in ~/Documents/chats - find file prices-a.md . read file and answer question"
    msg = user_message(InputText(text=question))
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)

    question = "review your answer to make sure you have not missed an important fact or been mislead by the question"
    msg = user_message(InputText(text=question))
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
    llm = LLM("gpt-4.1")
    # llm = LLM("gpt-4.1-mini", "use the eval tool as needed", [eval_fn], True)
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
        InputText(
            text="Task: mark the previous answer against the model answer. Award 1 mark for every correct inequality and 1 mark for every correct weight. \nModel answer:The 4 inequalities are S > C, T > 2C, 2S > T + C, 2C > S. Given all are natural numbers less than 10, the only solution is C=4, S=7, T=9"
        )
    )
    print_message(msg)
    response = llm.create(history, msg)
    print_response(response)


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
        response_first = client.responses.parse(model="gpt-4.1-nano", input=[user_message(InputText(text=question)).model_dump()], text_format=AnswerSheet)

        # Convert returned json to YAML format using process_response_output
        yaml_ans = process_response_output(response_first.output)

        # Replace the placeholder "{input}" in the answer template with the YAML output
        inp = answer_template.replace("{input}", yaml_ans)
        console.print(Markdown(inp), style="green")

        # Second parse: pass the updated answer input to "gpt-4.1-mini" model with MarkSheet format
        response_final = client.responses.parse(model="gpt-4.1-mini", input=[user_message(InputText(text=inp)).model_dump()], text_format=MarkSheet)

        # Process and print the final output
        process_response_output(response_final.output)

    # Process q5
    ask_question_and_mark(questions_q5, answers_q5)

    # Process q6
    ask_question_and_mark(questions_q6, answers_q6)


def main():
    # simple_message()
    # test_file_inputs()
    # test_image_analysis()
    # test_solve_visual_maths_problem()
    # structured_output_message()
    # test_function_calling()
    # test_function_calling_powershell()
    test_function_calling_python()
    # test_pdf_analysis()


if __name__ == "__main__":
    main()
