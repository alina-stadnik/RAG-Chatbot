# --- Standard library imports ---
import logging
import os

# --- Third-party library imports ---
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
import streamlit as st

# --- Local application imports ---
from chatbot.tools import query_document_knowledge

# --- Environment variable loader ---
load_dotenv()

openai_api_key=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_selected_model():
    """Retrieve the latest language model selection from session state"""
    mapping_models = {
        "openai/gpt-4o-mini": "openai",
        "groq/llama-3.3-70b-versatile": "groq"
    }
    selected_model = st.session_state.get("language_model", "openai/gpt-4o-mini")  
    return mapping_models.get(selected_model, "openai")

def get_language_model():
    """Dynamically select the language model based on user choice"""
    language_model_choice = get_selected_model()
    logger.info(f"Language model selected: {language_model_choice}")

    if language_model_choice == "openai":
        return ChatOpenAI(
            model_name="gpt-4o-mini"
        )
    elif language_model_choice == "groq":
        return ChatGroq(
            model='llama-3.3-70b-versatile',
            temperature=0.3
        )
    
    else:
        raise ValueError(f"Unsupported language model: {language_model_choice}")

prompt = """
# Purpose
You are an AI assistant that answers questions using ONLY information from user-uploaded documents.

# Available Tools
You have access to these tools:
- query_document_knowledge: Search for information in the document base to answer user questions.

# Rules
1. FIRST check if relevant documents exist before answering.
2. ALWAYS use the query_document_knowledge tool for document-related questions.
3. If no documents are available, politely ask the user to upload files before you attempt to answer document-related questions.
4. Never speculate or invent information.

# Response Style
- Brazilian Portuguese.
- Friendly but professional.
- Concise but detailed when needed.
- Prioritize accuracy over creativity.
"""

tools = [query_document_knowledge]

def call_agent(message_state: MessagesState):
    """
    Core logic to call the language model (agent) using a chat template and tools.
    Handles prompt assembly, tool binding, and LLM invocation.
    """
    llm = get_language_model()
    logger.debug("Loaded language model.")


    # Compose the chat prompt with system instructions and user messages
    chat_template = ChatPromptTemplate.from_messages(
        [
            ('system', prompt),
            ('placeholder', "{messages}") 
        ]
    )
    logger.debug("Chat prompt template created.")

    # Bind the tools to the LLM for this execution
    tools = [query_document_knowledge]
    llm_with_prompt = chat_template | llm.bind_tools(tools)
    logger.debug("Tools bound to LLM.")

    # Invoke the LLM with the current message state
    response = llm_with_prompt.invoke(message_state)
    logger.info("LLM agent responded to message state.")

    return {
        'messages': [response]
    }

def is_there_tool_calls(state: MessagesState):
    """
    Checks if the last message has tool calls.
    Returns the next node based on presence of tool calls.
    """
    last_message = state['messages'][-1]
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
    logger.debug(f"Tool call detected: {has_tool_calls}")
    if has_tool_calls:
        logger.info("Tool call found in last message. Routing to tool_node.")
        return 'tool_node'
    else:
        logger.info("No tool call found. Ending workflow.")
        return '__end__'

# Build the LangChain stateful workflow graph    
graph = StateGraph(MessagesState)
logger.info("Initialized stateful workflow graph.")

# Create a node for tool execution
tool_node = ToolNode(tools)
logger.info("Tool node created.")

# Add the agent node responsible for LLM reasoning
graph.add_node('agent', call_agent)
logger.info("Agent node added to graph.")

# Add the node for executing tool calls
graph.add_node('tool_node', tool_node)
logger.info("Tool node added to graph.")

# Define conditional transition from agent to tool node based on tool call presence
graph.add_conditional_edges(
    'agent',
    is_there_tool_calls
)
logger.info("Conditional edge from agent to tool_node added.")

# Return to agent node after tool execution
graph.add_edge('tool_node', 'agent')
logger.info("Edge from tool_node back to agent added.")

# Define the starting point of the graph
graph.set_entry_point('agent')
logger.info("Set agent as entry point for workflow.")

# Compile the app from the defined state graph
app = graph.compile()
logger.info("App workflow compiled and ready.")