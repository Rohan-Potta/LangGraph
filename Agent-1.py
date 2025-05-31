# Simple Bot => Objective is to implement llms into the graph
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
# In LangChain, HumanMessage is a class used to represent a message sent by a human in a conversation, typically as part of a chat interaction with a language model.
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv 
#env file is used to store secrets and keys and stuff.


load_dotenv()
class AgentState(TypedDict):
    messages : List[HumanMessage]

llm = ChatOpenAI(model='gpt-4o')

def process(state : AgentState)->AgentState:
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    return state

graph=StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent=graph.compile()


# user_input=input("Enter:")
# agent.invoke({'messages':[HumanMessage(content=user_input)]})


#now making it like a chatbot 
user_input=input("Enter: ")
while user_input!="exit":
    agent.invoke({'messages':[HumanMessage(content=user_input)]})
    user_input=input("Enter: ")

#now the biggest problem with this is that it doesnt have any memory allocated to it , for example if i just say my name is bob and ask who am I it wont be able to tell that it is bob  
# the main reason is that the api calls were independent that why 
