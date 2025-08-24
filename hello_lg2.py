from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
import re
import cohere
import os
# Load environment variables
from dotenv import load_dotenv 
load_dotenv()

# ----------------------------
# 1. Cohere client setup
# ----------------------------

co = cohere.Client( os.getenv("COHERE_API_KEY"))

# ----------------------------
# 2. State structure
# ----------------------------
class State(TypedDict, total=False):
    user_input: str
    llm_response: str
    result: str

# ----------------------------
# 3. Tools
# ----------------------------
@tool
def add_two_numbers(text: str) -> str:
    """Add the first two numbers found in the text."""
    nums = re.findall(r"\b\d+\b", text)
    if len(nums) >= 2:
        a, b = map(int, nums[:2])
        return f"The sum of {a} and {b} is {a + b}."
    return "I couldn't find two numbers to add."

@tool
def subtract_two_numbers(text: str) -> str:
    """Subtract the second number from the first number found in the text."""
    nums = re.findall(r"\b\d+\b", text)
    if len(nums) >= 2:
        a, b = map(int, nums[:2])
        return f"The difference when subtracting {b} from {a} is {a - b}."
    return "I couldn't find two numbers to subtract."

@tool
def multiply_two_numbers(text: str) -> str:
    """Multiply the first two numbers found in the text."""
    nums = re.findall(r"\b\d+\b", text)
    if len(nums) >= 2:
        a, b = map(int, nums[:2])
        return f"The product of {a} and {b} is {a * b}."
    return "I couldn't find two numbers to multiply."

# ----------------------------
# 4. LLM node using Cohere Chat API
# ----------------------------
def interpret(state: State) -> State:
    user_text = state.get("user_input", "")

    prompt = (
        f"You receive a user command: '{user_text}'.\n"
        "If it's asking to add numbers or sum or total, respond only with 'ADD'.\n"
        "If it's asking to subtract numbers or difference, respond only with 'SUBTRACT'.\n"
        "If it's asking to multiply numbers or product, respond only with 'MULTIPLY'.\n"
        "If it's none of the above, respond only with 'ECHO'."
    )

    response = co.chat(
        model="command-r",
        message=prompt
    )

    text = response.text.strip().upper()
    print("DEBUG LLM response:", text)
    return {"llm_response": text}

# ----------------------------
# 5. Tool nodes
# ----------------------------
def call_add_tool(state: State) -> State:
    text = state.get("user_input", "")
    result = add_two_numbers.invoke({"text": text})
    return {"result": result}

def call_subtract_tool(state: State) -> State:
    text = state.get("user_input", "")
    result = subtract_two_numbers.invoke({"text": text})
    return {"result": result}

def call_multiply_tool(state: State) -> State:
    text = state.get("user_input", "")
    result = multiply_two_numbers.invoke({"text": text})
    return {"result": result}

# ----------------------------
# 6. Echo node
# ----------------------------
def echo(state: State) -> State:
    return {"result": f"You said: {state.get('user_input', '')}"}

# ----------------------------
# 7. Build the LangGraph
# ----------------------------
builder = StateGraph(State)
builder.add_node("interpret", interpret)
builder.add_node("add_tool", call_add_tool)
builder.add_node("subtract_tool", call_subtract_tool)
builder.add_node("multiply_tool", call_multiply_tool)
builder.add_node("echo", echo)

builder.add_edge(START, "interpret")

# Conditional branching
def decide(state: State) -> str:
    resp = state.get("llm_response", "")
    if resp.startswith("ADD"):
        return "add_tool"
    elif resp.startswith("SUBTRACT"):
        return "subtract_tool"
    elif resp.startswith("MULTIPLY"):
        return "multiply_tool"
    else:
        return "echo"

builder.add_conditional_edges("interpret", decide)

builder.add_edge("add_tool", END)
builder.add_edge("subtract_tool", END)
builder.add_edge("multiply_tool", END)
builder.add_edge("echo", END)

graph = builder.compile()
print(graph)

# ----------------------------
# 8. Simple CLI loop
# ----------------------------
if __name__ == "__main__":
    while True:
        user = input("You: ")
        if user.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        state = {"user_input": user}
        new_state = graph.invoke(state)
        print("Bot:", new_state.get("result", "…"))
