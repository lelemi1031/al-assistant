from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.prompts import PromptTemplate
from langchain import hub

from langchain.agents import AgentExecutor

from utils import init_llm
from langchain_cohere.chat_models import ChatCohere

from langchain_community.tools.shell.tool import ShellTool
from langchain_experimental.llm_bash.bash import BashProcess

from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.tools import tool

from operator import itemgetter
from functools import partial


load_dotenv()


class PythonToolInput(BaseModel):
   code: str = Field(description="Python code to execute.")


class BashToolInput(BaseModel):
   code: str = Field(description="Bash command to execute.")

   
def get_bash_tool():
    bash = BashProcess()
    bash_tool = Tool(
    name="bash_tool",
    description="Executes bash command and returns the result. The code runs in astatic sandbox without interactive mode, so print output or save output to a file.",
    func=bash.run,
    )
    bash_tool.args_schema = BashToolInput

    return bash_tool


def get_python_tool():
    python = PythonREPL()
    python_tool = Tool(
    name="python_repl",
    description="Executes python code and returns the result. The code runs in astatic sandbox without interactive mode, so print output or save output to a file.",
    func=python.run,
    )
    python_tool.args_schema = PythonToolInput

    return python_tool

def get_shell_tool():
    shell_repl  = ShellTool()
    shell_tool = Tool(
    name="shell_repl",
    description=shell_repl.description + f"args {shell_repl.args}".replace(
    "{", "{{").replace("}", "}}"),
    func=shell_repl.run,
    )
    shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{").replace("}", "}}")

    return shell_tool


# @tool
# def get_files_tool():
#     """Get the list of files in the current directory."""
#     import subprocess
#     subprocess.run(["ls", "-l"]) 

# get_files_tool.name = "get_files" # use python case
# get_files_tool.description = "Get the list of files in the current directory"

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    print(first_int * second_int)
    return first_int * second_int

def tool_chain(model_output):
    tools = [multiply, add]
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    print(f"Chosen tool: {chosen_tool}")
    return itemgetter("arguments") | chosen_tool


def clean_text(text):
    return text.strip().replace("System: ", "").replace("?", "").replace("Assistant: ", "")#.replace("\n", " ")


def get_custom_agent(tools):
    from langchain.tools.render import render_text_description
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.runnables import RunnablePassthrough

    model = init_llm()
    
    rendered_tools = render_text_description(tools)
       

    system_prompt = f"""SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -

    {rendered_tools}

    The output needs to be in the following format:

    {{

        'name': <function name>,

        'arguments': <arguments to pass to the function>

    }}

    """

    system_prompt = f""" [INST] You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

    {rendered_tools}

    Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys.

    ONLY answer with the specified JSON format, no other text.
    [/INST]
    """
    

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    chain = prompt | model | print #clean_text | JsonOutputParser() | tool_chain #RunnablePassthrough.assign(output=partial(tool_chain, tools=tools))
    
    return chain



def init_agent(additional_tools: list = [], model=None):

    search = TavilySearchResults(name="tavily_search", description="Returns a list of relevant document snippets for a textual query retrieved from the internet.")
    tools = [search, ] + additional_tools

    if model == 'cohere':
        llm = ChatCohere(temperature=0.8, max_tokens=2048)
        prompt = hub.pull("hwchase17/openai-functions-agent")  
        agent = create_tool_calling_agent(llm, tools, prompt)
    else:
        agent = get_custom_agent(additional_tools)
    
    
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    message_history = ChatMessageHistory()

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # This is needed because in most real world scenarios, a session id is needed
        # It isn't really used here because we are using a simple in memory ChatMessageHistory
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_chat_history

@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    print(first_int + second_int)
    return first_int + second_int

additional_tools = [multiply, add]
agent_chat = get_custom_agent(additional_tools)

while True:
    question = input("Please enter the question: ")
    

    # try:
    agent_chat.invoke(
        {"input": question},
    )
    # except:
    #     pass


# if __name__ == '__main__':
#     additional_tools = [get_shell_tool(), get_python_tool()]
#     agent_chat = init_agent(additional_tools, model='cohere')

#     while True:
#         question = input("Please enter the question: ")
        

#         try:
#             agent_chat.invoke(
#                 {"input": question},
#                 # This is needed because in most real world scenarios, a session id is needed
#                 # It isn't really used here because we are using a simple in memory ChatMessageHistory
#                 config={"configurable": {"session_id": "<foo>"}},
#             )
#         except:
#             pass


# commands_examples = [
#     # "hi! I'm bob"
#     # "open the folder models in mac finder.",
#     "open safari using the command line.",
#     "I have a=10 and b=20. print the sum of a and b in python",
#     # "what's date today in toronto?",
#     ]

