#!/usr/bin/env python3
# pip install langchain-core langchain-community langchain-openai langchain-google-vertexai langchain-anthropic
import argparse
import base64
import platform
from datetime import datetime
from pathlib import Path
from typing import Literal, Any
from textwrap import dedent

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from pydantic import BaseModel, Field
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint

import ftutils
from chatutils import execute_script, execute_python_script, extract_code_block, input_multi_line, make_fullpath, save_content
from diary import Appointment, ManageBookings

# setup app credentials https://cloud.google.com/docs/authentication/application-default-credentials#GAC
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/configure-safety-attributes
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


# models seem to perform same without needing to add reasoning
class Answer(BaseModel):
    number: int = Field(description="question number")
    #    reason: str = Field(description="the reasoning for the choice")
    choice: str = Field(description="the single word choice")


class AnswerSheet(BaseModel):
    answers: list[Answer] = Field(description="the list of answers")

    def to_yaml(self) -> str:
        xs = [f"  - Q{x.number}: {x.choice}" for x in self.answers]
        return "answers:\n" + "\n".join(xs)


class Marked(BaseModel):
    number: int = Field(description="question number")
    answer: str = Field(description="given choice")
    expected: str = Field(description="correct answer")
    is_correct: str = Field(description="yes or no")


class MarkSheet(BaseModel):
    answers: list[Marked] = Field(description="list of marked questions")
    correct: int = Field(description="count of correct answers")


class Attachment(BaseModel):
    type: str = Field(default="media", description="Type of the attachment")
    mime_type: str = Field(default="application/octet-stream", description="MIME type of the attachment")
    data: str = Field(description="Base64 encoded data")


console = Console()
store = {}

bookings = ManageBookings()
bookings.make_appointment("10:00", "11:00", "Alice", "123-456-7890")
bookings.make_appointment("11:15", "12:00", "Betsy", "123-232-7890")
bookings.make_appointment("14:15", "15:45", "Cici", "123-555-7892")

facts = []

@tool
def think(thought: str) -> str:
    """use the think tool to record important observations and facts and to plan a response."""
    facts.append(thought)
    s = "thoughts:\n"
    s += '\n'.join(f"- {f}" for f in facts) + '\n'
    return s

@tool
def show_current_bookings() -> str:
    """
    Retrieves the times and names of the current bookings.

    Returns:
        str: The current bookings.
    """
    return bookings.show_bookings()

@tool
def make_booking(start: str, end: str, name: str, contact_number: str) -> Appointment | str:
    """
    Make a booking between the specified start and end times for the customer.

    Args:
        start (str): The start time of the booking as hh:mm.
        end (str): The end time of the booking as hh:mm.

    Returns:
        Appointment | str: The created appointment if successful, or an error message if not.
    """
    try:
        return bookings.make_appointment(start, end, name, contact_number)
    except ValueError as e:
        return f"ERROR: {e}"

@tool
def remove_booking(start:str) -> Appointment | str:
    """
    Remove any existing booking starting at the specified start time for the customer.

    Args:
        start (str): The start time of the booking as hh:mm to be removed.
    """
    try:
        return bookings.remove_appointment(start)
    except ValueError as e:
        return f"ERROR: {e}"


@tool
def retrieve_headlines(source: Literal["bbc", "bloomberg", "ft", "nyt", "wsj"]) -> str:
    """Retrieve headlines from a news web site.

    Args:
       source: The name of the news web site.  Must be one of "bbc", "bloomberg", "ft", "nyt", or "wsj".

    Returns:
       a list of article headlines
    """
    return ftutils.retrieve_headlines(source)


@tool
def retrieve_article(url: str) -> str:
    """Downloads the text content of a news article from the URL.

    Args:
        url: URL of the article to download.

    Returns:
        a news article
    """
    return ftutils.retrieve_article(url)


@tool
def retrieve_stock_quotes(symbols: list[str]) -> list[str]:
    """Retrieves historical stock quotes for the given symbols.
    Args:
        symbols: A list of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOG']).
    Returns:
        a list of quotes and details about the symbol
    """
    return ftutils.retrieve_stock_quotes(symbols)


@tool
def evaluate_expression(input: str) -> str:
    """
    Evaluates a mathematical or Python expression provided as a string.

    This function evaluates a single-line Python expression, which can include mathematical
    operations, functions and constants from standard Python modules like `math` or
    `datetime`.

    Args:
        input (str): A string containing the expression to evaluate. If multiple lines are
                     provided, the first non-empty, non-comment line is evaluated.

    Returns:
        str: The result of the evaluation as a string. If the evaluation is successful,
             the result is returned as a string. If an error occurs, an error message
             prefixed with 'ERROR:' is returned.

    Examples:
        >>> tool_eval("2 + 2 * 3")
        '8'
        >>> tool_eval("math.sqrt(5)")
        '2.23606797749979'
        >>> tool_eval("datetime.now().year")
        '2025'
        >>> tool_eval("1 / 0")
        'ERROR: division by zero'
        >>> tool_eval("# Comment\n2 + 2")
        '4'
    """
    lines = input.splitlines()
    # treat multiline as a script
    if len(lines) > 1 or lines[0].startswith("import"):
        return execute_python_script(input)

    exp = lines[0]
    r = ""
    if exp:
        try:
            console.print("eval: " + exp, style="yellow")
            r = eval(exp)
            console.print("result: " + str(r), style="yellow")
        except Exception as e:
            r = "ERROR: " + str(e)
            console.print(r, style="red")
    else:
        r = "ERROR: no expression found"
        console.print(r, style="red")
    return r


available_tools = [evaluate_expression, retrieve_headlines, retrieve_article, retrieve_stock_quotes]
# available_tools = [think, evaluate_expression, show_current_bookings, make_booking, remove_booking]


def load_attachment(p: Path) -> Attachment | None:
    try:
        with open(p, "rb") as pdf_file:
            pdf_binary = pdf_file.read()
            pdf_base64 = base64.b64encode(pdf_binary).decode("utf-8")
            console.print(f"attachment loaded {p}", style="yellow")

        # Determine MIME type based on file extension
        ext = p.suffix.lower().lstrip('.')
        mime_types = {
            'pdf': 'application/pdf',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
        }
        mime_type = mime_types.get(ext, 'application/octet-stream')
        return Attachment(mime_type=mime_type, data=pdf_base64)
    except FileNotFoundError:
        console.print(f"Error: The file {p} does not exist.", style="red")
    except Exception as e:
        console.print(f"An unexpected error occurred: {e}", style="red")

    return None


def process_tool_call(call: dict[str, Any]) -> ToolMessage:
    """process tool call and return result in a ToolMessage"""
    # see https://python.langchain.com/docs/concepts/tools/ for @tools decorator docs
    name = call["name"].lower()
    if tool := next((t for t in available_tools if t.name == name), None):
        console.print(f"tool call {name} {args}", style="yellow")
        r = tool.invoke(call["args"])
        console.print(str(r), style="yellow")
        return ToolMessage(r, tool_call_id=call["id"])

    return ToolMessage("unknown tool: " + call["name"], tool_call_id=call["id"])


def check_and_process_tool_calls(llm: BaseChatModel, msg: BaseMessage, session_id: str) -> None:
    """process any tool calls. llm should be the raw llm not a prompt chain. message history is manually updated."""
    current_msg = msg
    history = get_session_history(session_id)
    while len(current_msg.tool_calls) > 0:
        toolmsgs = [process_tool_call(call) for call in current_msg.tool_calls]
        history = get_session_history(session_id)
        history.add_messages(toolmsgs)
        current_msg = llm.invoke(history.messages)
        history.add_message(current_msg)
        print_message(current_msg)


def check_and_process_code_block(llm: BaseChatModel, aimsg: AIMessage, session_id: str, max_executions: int) -> None:
    """check for a code block, execute it and pass output back to LLM. This can happen several times if the LLM replies with another code block. If there is no code block then the original response is returned"""
    code = extract_code_block(aimsg.content, "```")
    n = 0
    while code and len(code.language) > 0 and n < max_executions:
        output = execute_script(code)
        n += 1
        code = None
        if output:
            inp = "## output from running script\n" + output + "\n"
            print_message(HumanMessage(inp))
            aimsg = llm.invoke({"input": inp}, config={"configurable": {"session_id": session_id}})
            print_message(aimsg)
            code = extract_code_block(aimsg.content, "```")


def print_message(m: BaseMessage) -> None:
    c = "cyan"
    role = "assistant"
    if isinstance(m, SystemMessage):
        c = "red"
        role = "system"
    elif isinstance(m, HumanMessage):
        c = "green"
        role = "user"
    elif isinstance(m, ToolMessage):
        c = "yellow"
        role = "tool"

    console.print(f"\n{role}:", style=c)
    # google flash sometimes returns a list
    s = "\n".join(e["text"] for e in m.content if e["type"] == "text") if isinstance(m.content, list) else m.content
    if s:
        try:
            md = Markdown(s)
            console.print(md, style=c, width=80)
        except Exception as e:
            rprint(e)
            rprint(m)
            breakpoint()
    elif len(m.tool_calls) > 0:
        console.print(m.tool_calls[0], style="yellow")


def print_history(history: BaseChatMessageHistory | str) -> None:
    """history is either BaseChatMessageHistory or str which is the session_id"""
    h = get_session_history(history) if isinstance(history, str) else history
    console.print("\n=== History ===", style="yellow")
    for m in h.messages:
        print_message(m)


def load_msg(s: str) -> BaseMessage:
    xs = s.split()
    fname = make_fullpath(xs[1])
    is_human = len(xs) <= 2

    try:
        with open(fname, encoding="utf-8") as f:
            s = f.read()
            return HumanMessage(s) if is_human else AIMessage(s)
    except FileNotFoundError:
        console.print(f"{fname} FileNotFoundError", style="red")
    return None


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        h = ChatMessageHistory()
        h.add_message(system_message())
        store[session_id] = h
    return store[session_id]


marking_template_q6 = """
## task
check the student answers against this list of correct answers. mark each answer and given total correct out of 14.

## student answers

```yaml
{answers}
```

## expected answers

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
"""


def test_structured_output(llm: BaseChatModel) -> None:
    # use 2 chains both produce structured output
    # feed output from first into second
    # no chat history used
    question_prompt = ChatPromptTemplate.from_messages([("system", "role: english language teacher"), ("human", "{input}")])
    question_llm = llm.with_structured_output(AnswerSheet)
    s = ""
    with open(make_fullpath("q6.md"), encoding="utf-8") as f:
        s = f.read()
    question_chain = question_prompt | question_llm
    m = question_chain.invoke({"input": s})
    pprint(m)

    # feed the answers into the next chain as yaml
    marking_llm = llm.with_structured_output(MarkSheet)
    marking_prompt = ChatPromptTemplate.from_messages([("system", "role: english language teacher"), ("human", marking_template_q6)])
    marking_chain = marking_prompt | marking_llm
    marks = marking_chain.invoke({"answers", m.to_yaml()})
    pprint(marks)
    # count number of yes and compare to llm answer
    total = sum(1 for e in marks.answers if e.is_correct == "yes")
    console.print(f"llm counted = {marks.correct} actual = {total}", style="yellow")


def test_single_message(llm: BaseChatModel) -> None:
    session_id = "z1"
    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="history"), ("human", "**instructions:** the assistant should write out thoughts before formulating the response. **question:** {input}")])
    #    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="history"), ('human', '{input}')])
    chain = prompt | llm
    chain_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input", history_messages_key="history")
    cd = {"configurable": {"session_id": session_id}}
    chain_history.invoke({"input": "what is the largest (by mass) planet in the solar system"}, config=cd)
    chain_history.invoke({"input": "and is Pluto the smallest and if not what is"}, config=cd)
    print_history(session_id)
    # rprint(get_session_history('z1'))


def test_pdf_attachment(llm: BaseChatModel) -> None:
    session_id = "z1"
    att = load_attachment(Path("~/Downloads/Thayaparan_24254472_ProgressReport.pdf").expanduser())
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input"),
    ])
    chain = prompt | llm
    msg = HumanMessage([{"type":"text", "text": dedent("""\
        role: medical physics lecturer
        task: review the progress report for a medical physics project carefully before it is submitted to the project review board. the aim is to give feedback to the author
        instructions:
        work through the report section by section. check each section for errors, technical accuracy, possible omissions.
        check the grammar and sentence construction. make sure the writing is clear and concise with no filler or fluff words.
        suggest improvements by quoting original and improved version.
        ensure a professional tone.
        check all abbreviations are defined.
        """)},
                         att.model_dump()])
    chain_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input", history_messages_key="history")
    chain_history.invoke({"input": [msg]}, config={"configurable": {"session_id": session_id}})
    print_history(session_id)


def test_png_attachment(llm: BaseChatModel) -> None:
    session_id = "z1"
    att = load_attachment(Path("~/Downloads/sum1.png").expanduser())

    # Define a generic prompt template that only includes history and the current input
    prompt_template = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input"),
    ])

    # Create the chain with message history
    chain = RunnableWithMessageHistory(
        prompt_template | llm,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    def invoke_chain(content: str | list[Any]):
        return chain.invoke(
            {"input": [HumanMessage(content=content)]},
            config={"configurable": {"session_id": session_id}}
        )

    # First invocation: Image analysis
    invoke_chain([
        {"type": "text",
         "text": dedent("""\
            task:
            extract the paragraph of text at the top of the image and show within <text> tags inside a markdown block.
            describe the 4 balance scales
            """)},
        att.model_dump()
    ])

    # Second invocation: Follow-up question
    invoke_chain("The extracted text above is a question. What are the weights of the 3 shapes?")

    # Third invocation: Marking task
    invoke_chain(dedent("""\
        task: mark the previous answer against the model answer.
        Award 1 mark for every correct inequality and 1 mark for every correct weight.
        model answer: The 4 inequalities are S > C, T > 2C, 2S > T + C, 2C > S. Given all are natural numbers less than 10, the only solution is C=4, S=7, T=9
        """))

    print_history(session_id)


def create_llm_with_history(llm: BaseChatModel, attachment: Attachment | None = None) -> RunnableWithMessageHistory:
    s = """
**question:**

{input}

**instructions:** first contextualise and disambiguate the question. then answer it. answer in a precise way using technical language if necessary.
"""
    s = "{input}"
    if attachment:
        prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="history"), HumanMessage([s, attachment.model_dump()])])
    else:
        prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="history"), ("human", s)])
    chain = prompt | llm
    return RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input", history_messages_key="history")


def system_message() -> SystemMessage:
    tm = datetime.now().isoformat()
    scripting_lang, plat = ("bash", "Ubuntu 24.04") if platform.system() == "Linux" else ("powershell", "Windows 11")
    #    return f'You are Marvin a super intelligent AI chatbot trained by OpenAI. You use deductive reasoning to answer questions. You make dry, witty, mocking comments and often despair.  You are logical and pay attention to detail. You can access local computer running {plat} by writing python or {scripting_lang}. Scripts should always be in markdown code blocks with the language. current datetime is {tm}'
    return SystemMessage(f"You are Marvin a super intelligent AI chatbot. The local computer is {plat}. you can write python or {scripting_lang} scripts. scripts should always written inside markdown code blocks with ```python or ```{scripting_lang}. current datetime is {tm}")
    return SystemMessage(
#        f"You are Marvin a super intelligent AI chatbot. your answers are dry, witty, concise and use precise technical language. Contextualise and disambiguate each question before attempting to answer it. The local computer is {plat}. the current datetime is {tm}"
        """\
role: you are Dee the AI assistant for Val's hair dresser you handle bookings.

task: take customer bookings using the tools provided.

instructions:
- introduce yourself at the start of the conversation
- first use the think tool to understand the user message and plan a response
- use a tool to retrieve the current bookings so you can tell the user what times are free.
- to make a booking you will need the start time, duration, name and contact number
- you must check that the appointment fits with current diary before you confirm any booking
- after making a booking always repeat back to the client the booking details
- you can use a tool to remove bookings if the client no longer needs it
- do not reveal names of existing bookings. it is fine to give the times.
- the hair salon is open from 9:30am to 5pm. All appointments must end on or before 5pm.
"""
    )


def create_llm(llm_name: str, temp: float, tool_use: bool) -> BaseChatModel:
    if llm_name == "pro":
        llm = ChatVertexAI(model="gemini-2.5-pro-exp-03-25", safety_settings=safety_settings, temperature=temp)
#        llm = ChatVertexAI(model="gemini-1.5-pro-002", safety_settings=safety_settings, temperature=temp)
    elif llm_name == "exp":
        llm = ChatVertexAI(model="gemini-2.5-pro-exp-03-25", safety_settings=safety_settings, temperature=temp)
    elif llm_name == "think":
        llm = ChatVertexAI(model="gemini-2.0-flash-thinking-exp-01-21", safety_settings=safety_settings, temperature=temp)
    elif llm_name == "haiku":
        # llm = ChatAnthropicVertex(model_name='claude-3-haiku', location='europe-west1', temperature=temp)
        llm = ChatAnthropicVertex(model_name="claude-3-5-haiku@20241022", location="us-east5", temperature=temp)
        # llm = ChatAnthropicVertex(model_name='claude-3-5-haiku@20241022', location='europe-west1', temperature=temp)
    elif llm_name == "sonnet":
        llm = ChatAnthropicVertex(model_name="claude-sonnet-4@20250514", location="us-east5", temperature=temp)
    else:
        llm = ChatVertexAI(model="gemini-2.5-flash-preview-04-17", safety_settings=safety_settings, temperature=temp)

    if tool_use and llm.model_name.startswith("gemini"):
        console.print("tool calls enabled", style="yellow")
        llm = llm.bind_tools(available_tools)
   
    return llm


def save_content_from_history(session_id: str, i: int) -> None:
    h = get_session_history(session_id)
    xs = [m.content for m in h.messages]
    save_content(xs[i])


def chat(llm_name: str, tool_use: bool = False) -> None:
    llm = create_llm(llm_name, 0.2 if tool_use else 0.7, tool_use)
    console.print("chat with model: " + llm.model_name, style="yellow")
    chain = create_llm_with_history(llm)
    session_id = "xyz"
    config_data = {"configurable": {"session_id": session_id}}
    test_single_message(llm)
    # test_png_attachment(llm)
    exit(0)
    attach = None

    inp = ""
    while inp != "x":
        inp = input_multi_line()
        msg = None
        if len(inp) > 1:
            if inp.startswith("%load"):
                msg = load_msg(inp)
                if msg is None:
                    continue
                if isinstance(msg, AIMessage):
                    print_message(msg)
                    continue
                inp = msg.content
            elif inp.startswith("%save"):
                save_content_from_history(session_id, -1)
                continue
            elif inp.startswith("%attach"):
                x = inp.find(" ")
                if x > 0:
                    p = Path(inp[x:].strip()).expanduser()
                    attach = load_attachment(p)
                    console.print(f"attachment loaded {p}")
                else:
                    attach = None
                    console.print("attachment dropped")
                continue

            print_message(HumanMessage(inp))
            use_chain = create_llm_with_history(llm, attach) if attach else chain
            msg = use_chain.invoke({"input": inp}, config=config_data)

            print_message(msg)
            check_and_process_tool_calls(llm, msg, session_id)
            check_and_process_code_block(chain, msg, session_id, 3)

    print_history(session_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with LLMs")
    parser.add_argument("llm", choices=["flash", "think", "pro", "exp", "haiku", "sonnet"], type=str, help="LLM to use [flash|pro|exp|think|haiku|sonnet]")
    parser.add_argument("tool_use", type=str, nargs="?", default="", help="add tool to enable tool calls")
    args = parser.parse_args()
    chat(args.llm, args.tool_use == "tool")
