"""
ReAct Agent -> Reasoning and Acting Agent

The point is that , there is Start-> Agent -> Tool (Looped to Agent and back to tool) ->(Coming from the Agent Node) END

+--------+     +--------+     +------+
| START  | --> | Agent  | --> | END  |
+--------+     +--------+     +------+
                   | ^
                   v |
              +---------+
              |  Tools  |
              +---------+
This is ReAct Agent as the tools always goes back to the agent and not the end

The llm decides how the tool does the work and till when is it needed 
"""
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage  # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage  # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage  # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

"""
Annotated => gives more context to the variable without affecting the variable type, for example email that needs like abc@email.com.
email = Annotated[str,"This has to be a valid email format]
print(email.__metadata__)
email = Annotated[str,"This has to be a valid email format"]
print(email.__metadata__)
#Output =>  ('This has to be a valid email format',)


Sequence => TO automatically handle the state updates for sequences such as by adding new messages to a chat history 
This basically exists to prevent any list manipulation of the graph nodes 

ToolMessage is like the parent class to the HumanMessage AIMessage which all act as the child class

from langchain_core.tools import tools
In LangChain, tools are functions or operations the LLM can call to perform actions — like searching the web, calling an API, doing math, or interacting with a database.


from langgraph.prebuilt import ToolNode

ToolNode is a prebuilt LangGraph node designed to handle tool usage automatically.
In LangGraph, a node is a step in the workflow graph. ToolNode:
Takes user input
Determines which tool to call
Passes input to the correct tool
Returns the result back as a ToolMessage
So it's like a "smart agent step" that manages tool usage for you.

from langgraph.graph.message import add_messages
This is a reducer function 
    This is a rule that controls how updates from a nodes are combined with an exisiting state 
    Tells us how to merge new data into the current state
    Without a reducer , updates would be replaced entirely with the existing value 

# --- Without a reducer ---
state = {"messages": ["Hi!"]}
update = {"messages": ["Nice to meet you!"]}
new_state = update
print("Without Reducer:", new_state)
# Output: {"messages": ["Nice to meet you!"]}


# --- With a reducer ---
state = {"messages": ["Hi!"]}
update = {"messages": ["Nice to meet you!"]}
new_state = {"messages": state["messages"] + update["messages"]}
print("With Reducer:", new_state)
# Output: {"messages": ["Hi!", "Nice to meet you!"]}

"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
            # What this means is that annotated makes sure it is in the right form, sequence to get the format as it is stored in metadata , BaseMessage is the parent class, add_message is the reducer funtion 


@tool
def add(a:int,b:int):
    """This is an addition function to add two numbers"""
    return a+b
@tool
def subtract (a:int,b:int):
    """This is a subtract function to add two numbers"""
    return a-b
@tool
def multiply(a:int,b:int):
    """This is a multiply function to add two numbers"""
    return a*b


tools = [add,subtract,multiply] #The doc string is needed else without it the llm wont know what the tool has to do  

model = ChatOpenAI(model='gpt-4o').bind_tools(tools) #this bind tools command is done so that the model, will understand what all tools it can access

def model_call (state:AgentState)->AgentState:
    system_promt = SystemMessage(content= "You are my AI Assistant, please answer my query to the best of your ability.")
    response=model.invoke([system_promt]+state["messages"]) #The state messages is used so that the query given by the user is also added to the model response 
    return {'messages':[response]} #what this does is that it updates message with the response 

"""
| Message Type    | Represents...                      | Role in Conversation                                      | Who Sends It     |
| --------------- | ---------------------------------- | --------------------------------------------------------- | ---------------- |
| `SystemMessage` | Instructions or context for the AI | Sets behavior, tone, rules (e.g. "You are a helpful bot") | System/Developer |
| `HumanMessage`  | Input from the user                | Natural language questions or commands                    | Human (user)     |
| `AIMessage`     | Response from the AI               | The AI model's reply based on the context                 | AI model         |

"""


def should_continue(state: AgentState):
    messages=state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: #this will basically check if there is any tools calls still left within the tools list that the node has to complete
        return "end"
    else:
        return "continue"
    
# defining the graph
graph =StateGraph(AgentState)
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools=tools) 
graph.add_node("tools",tool_node) #this is to create a 
graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END
    }
)

graph.add_edge("tools","our_agent") #this is done to re-route from the node to the agent 

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs ={"messages":[("user","add 40 + 12 and then multiply the result by 5 and then tell me a joke as well")]}
print_stream(app.stream(inputs,stream_mode="values"))

"""
python .\Agent-3.py
================================ Human Message =================================

add 40 + 12 and then multiply the result by 5 and then tell me a joke as well
================================== Ai Message ==================================
Tool Calls:
  add (call_RvIG7JOmK8TFUcxZQqNjwpLE)
 Call ID: call_RvIG7JOmK8TFUcxZQqNjwpLE
  Args:
    a: 40
    b: 12
  multiply (call_u69WW5dsIPb5NuZudGaXWpnc)
 Call ID: call_u69WW5dsIPb5NuZudGaXWpnc
  Args:
    a: 52
    b: 5
================================= Tool Message =================================
Name: multiply

260
================================== Ai Message ==================================

The result of 40 + 12 is 52, and when multiplied by 5, the result is 260.
Now, here's a joke for you:
Why don't scientists trust atoms?
Because they make up everything!

And now this is the power of llms , where they are so rohbust ,even tho we dont have the tool it can still answer us

"""

