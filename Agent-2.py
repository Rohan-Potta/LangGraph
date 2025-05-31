"""
The objective in this is to make a chatbot with conversational history
and this uses two types , Human messages and AIMessage

HumanMessage
Represents a message sent by the user (human).
AIMessage
Represents a message sent by the AI model in response to a user input.

Why Are These Important?
LangChain allows you to pass structured chat history to chat-based LLMs. You can construct a full conversation with message types like:
    from langchain.schema import HumanMessage, AIMessage

    chat_history = [
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi! How can I help you today?"),
        HumanMessage(content="What's the weather like in Paris?")
    ]
"""

import os
from typing import TypedDict, List,Union
"""
Union
Represents a variable that can be one of multiple types.
def get_id(val: Union[int, str]) -> str:
    return f"ID: {val}"
This function accepts either an int or a str as input.
"""
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv 

load_dotenv()


def load_conversation_history(file_path: str):
    history = []
    if not os.path.exists(file_path):
        return history
    
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("You: "):
                content = line.replace("You: ", "").strip()
                history.append(HumanMessage(content=content))
            elif line.startswith("AI: "):
                content = line.replace("AI: ", "").strip()
                history.append(AIMessage(content=content))
    return history

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]

llm =ChatOpenAI(model="gpt-4o")

def process(state: AgentState)->AgentState:
    """This node will solce the request of the input"""
    response=llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI:{response.content}")

    print("CURRENT STATE: ",state['messages'])
    return state

graph=StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent=graph.compile()

conversation_history = load_conversation_history("logging.txt")

user_input = input("Enter: ")
while user_input!="exit":
    conversation_history.append(HumanMessage(content=user_input))
    #We use the HumanMessage so that , the langchain can understand the wrapper around knows it is a human message 
    result = agent.invoke({'messages':conversation_history})

    print(result['messages'])

    conversation_history=result['messages']
    user_input = input("Enter: ")

#Now the major issue is that the moment i exit from the code the entire histroy is wiped out , so instead we write into a text file and read the context from there

with open("logging.txt","w") as file:
    file.write("This is the conversational log for agent-2\n")

    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n")
        else:
            file.write("End of the conversation")

print("Conversation has been saved to logging.txt")


#To prevent the conversation_history to get too long and keep on increasing in size what we can do is remove the last message from the conversation history when the size is like 5 or 10