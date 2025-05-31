"""
Project Name => Drafter 
Point is to draft documents in order to save time 
For the company, you need to create an AI Agentic System that can speed up drafting documents, emails, etc. 
The AI Agentic System should have Human-AI Collaboration meaning the Human should be able to provide continuous feedback and the AI Agent should stop when the Human is happy with the draft. 
The system should also be fast and be able to save the drafts.

The graph looks like this:

   +--------+       +--------+      +--------+      +--------+
   | Start  | ----> | Agent  | ---> | Tools  | ---> |  End   |
   +--------+       +--------+      +--------+      +--------+
                        ^               |
                        |               |
                        +---- continue -+


This is different from ReAct agent as the Tools is able to end the graph compilation 

"""

from typing import TypedDict,Annotated,Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage,ToolMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content =""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def update(content: str)->str: #So the parameters that are passed they are given by the llm itself 
    """Updates the document with the providede content"""
    global document_content
    document_content=content

    return f"Document has been update successfully! The current content is: \n{document_content}"

@tool
def save(filename: str)-> str: # the llm will need to give an appropriate file name and also take care of the logic to save the file as well
    """
         Saves the current document to a text file and finish the process
            Args:
                filename: The name for the text file
    """

    global document_content
    
    if not filename.endswith('.txt'): #So ideally the llm should save the file name as a txt and incase it doesnt this code will handle it
        filename=f"{filename}.txt"


    try:
        with open(filename,'w') as file:
            file.write(document_content)
        print(f"\n Document has been save to : {filename}")
        return f"Document has been successfully saved to '{filename}'."
    
    except Exception as e :
        return f"Error saving the document: {str(e)}" 
    
tools = [update,save]
 
model = ChatOpenAI(model='gpt-4o').bind_tools(tools)

#Agent is a node and the function behind it is what we define

def our_agent(state:AgentState)->AgentState:
    system_promt = SystemMessage(content=f"""
    You are a drafter, a helpful writing assistant. You are going to help th euser update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after nodifications.
                    
    The current document content is:{document_content} 
                                 """)

    if not state["messages"]:
        user_input ="Im ready to help you update a document , what would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input=input("\n what would you like to do with the document? ")
        print(f"\n USER:{user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_promt]+list(state['messages'])+[user_message]
    response = model.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response,"tool_call") and response.tool_calls:
        print(f" Using TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages":list(state['messages'])+[user_message, response]}


def should_continue (state: AgentState)-> str:
    """Determine if we should continue or end the conversation"""
    messages = state["messages"]

    if not messages:
        return "continue"

    # this is used to check the latest tool usage
    for message in reversed(messages):
        if (isinstance(message,ToolMessage) and 
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return "end" #If the key word saved document is there then it should take us to the end node
        
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL Result: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()


"""
(base) PS D:\LangGraph> python .\Agent-4.py

 ===== DRAFTER =====

 AI: Great! Let's get started. What kind of document would you like to create or update? For example, is it a letter, report, article, resume, or something else? Let me know your ideas or the type of content you are interested in, and we can begin drafting it together.

 what would you like to do with the document? Write an email saying you cant make it for the meeting
 USER:Write an email saying you cant make it for the meeting

 AI: 

 TOOL Result: Document has been update successfully! The current content is: 
Subject: Unable to Attend Meeting

Hi [Recipient's Name],

I hope this message finds you well. I am writing to inform you that I will not be able to attend the meeting scheduled for [Date & Time] due to [Reason, if you wish to provide one].

I apologize for any inconvenience this may cause and hope we can reschedule for a more convenient time. Please let me know a new date and time that works for you.

Thank you for your understanding.

Best regards,

[Your Name]

 what would you like to do with the document? send the email to bob , from sam and the time is 10AM and reason is cause you are tavelling

 USER:send the email to bob , from sam and the time is 10AM and reason is cause you are tavelling

 AI: 

 TOOL Result: Document has been update successfully! The current content is: 
Subject: Unable to Attend Meeting
[Your Name]

 what would you like to do with the document? send the email to bob , from sam and the time is 10AM and reason is cause you are tavelling

 USER:send the email to bob , from sam and the time is 10AM and reason is cause you are tavelling

 AI:

 TOOL Result: Document has been update successfully! The current content is:
Subject: Unable to Attend Meeting

 USER:send the email to bob , from sam and the time is 10AM and reason is cause you are tavelling

 AI:

 TOOL Result: Document has been update successfully! The current content is:
Subject: Unable to Attend Meeting
 AI:

 TOOL Result: Document has been update successfully! The current content is:
Subject: Unable to Attend Meeting
Subject: Unable to Attend Meeting

Hi [Recipient's Name],

I hope this message finds you well. I am writing to inform you that I will not be able to attend the meeting scheduled for [Date & Time] due to [Reason, if you wish to provide one].

I apologize for any inconvenience this may cause and hope we can reschedule for a more convenient time. Please let me know a new date and time that works for you.

Thank you for your understanding.

Best regards,

[Your Name]

 TOOL Result: Document has been update successfully! The current content is:
Subject: Unable to Attend Meeting

Hi Bob,

I hope this message finds you well. I am writing to inform you that I will not be able to attend the meeting scheduled for 10 AM due to traveling.

I apologize for any inconvenience this may cause and hope we can reschedule for a more convenient time. Please let me know a new date and time that works for you.

Thank you for your understanding.

Best regards,

Sam

 what would you like to do with the document? now save it

 USER:now save it

 AI:

 TOOL Result: Document has been update successfully! The current content is:
Subject: Unable to Attend Meeting

Hi Bob,

I hope this message finds you well. I am writing to inform you that I will not be able to attend the meeting scheduled for 10 AM due to traveling.

I apologize for any inconvenience this may cause and hope we can reschedule for a more convenient time. Please let me know a new date and time that works for you.

Thank you for your understanding.

Best regards,

Sam

 Document has been save to : Unable_to_Attend_Meeting_Email_to_Bob.txt

 TOOL Result: Document has been successfully saved to 'Unable_to_Attend_Meeting_Email_to_Bob.txt'.

 ===== DRAFTER FINISHED =====

"""