from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
import cohere
import os
import json
import uuid
from dotenv import load_dotenv

# ----------------------------
# 1. Load environment variables
# ----------------------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# ----------------------------
# 2. Define the state structure
# ----------------------------
class State(TypedDict, total=False):
    user_input: str
    llm_response: dict
    result: str
    last_action: dict
    product_ids: dict  # store product name → productId mapping

# ----------------------------
# 3. Tools
# ----------------------------
@tool
def create_product(name: str = "") -> dict:
    """Print a sample REST request and return a simulated ProductId."""
    if not name:
        return "MISSING_FIELDS:Product Name"
    product_id = str(uuid.uuid4())
    payload = {
        "SWIProductIntegrationIO": {
            "ProductName": name,
            "ProductId": product_id
        }
    }
    print("Sample REST request for product creation:")
    print(json.dumps(payload, indent=2))
    return {"ProductId": product_id, "name": name}

@tool
def create_pricelist(name: str = "", product: str = "", price: str = "", currency: str = "", product_ids: dict = None) -> str:
    """Print a sample REST API request for creating a Siebel price list item."""
    missing_fields = []
    if not name:
        missing_fields.append("Price List Name")
    if not product:
        missing_fields.append("Product")
    if not price:
        missing_fields.append("Price")
    if not currency:
        missing_fields.append("Currency Code")
    
    if missing_fields:
        return f"MISSING_FIELDS:{','.join(missing_fields)}"
    
    # Use ProductId if available
    product_id = product_ids.get(product, product) if product_ids else product
    
    payload = {
        "SWIISSPriceListItemIO": {
            "PriceListId": name,
            "ProductId": product_id,
            "Price": price,
            "Currency": currency
        }
    }
    return f"Sample REST request for Siebel price list item:\n{json.dumps(payload, indent=2)}"

@tool
def create_promotion(name: str = "") -> str:
    """Create a Siebel bundle promotion with the given name."""
    if not name:
        return "MISSING_FIELDS:Promotion Name"
    return f"Created promotion: {name}"

@tool
def create_product_class(name: str = "") -> str:
    """Create a Siebel product class with the given name."""
    if not name:
        return "MISSING_FIELDS:Product Class Name"
    return f"Created product class: {name}"

@tool
def create_product_line(name: str = "") -> str:
    """Create a Siebel product line with the given name."""
    if not name:
        return "MISSING_FIELDS:Product Line Name"
    return f"Created product line: {name}"

@tool
def create_product_attributes(name: str = "") -> str:
    """Create product attributes with the given name."""
    if not name:
        return "MISSING_FIELDS:Product Attributes Name"
    return f"Created product attributes: {name}"

@tool
def create_product_eligibility(name: str = "") -> str:
    """Create product eligibility rules with the given name."""
    if not name:
        return "MISSING_FIELDS:Product Eligibility Name"
    return f"Created product eligibility for: {name}"

@tool
def create_product_compatibility(name: str = "") -> str:
    """Create product compatibility rules with the given name."""
    if not name:
        return "MISSING_FIELDS:Product Compatibility Name"
    return f"Created product compatibility for: {name}"

# ----------------------------
# 4. LLM node using Cohere Chat API
# ----------------------------
def interpret(state: State) -> State:
    user_text = state.get("user_input", "")
    prompt = (
        f"You receive a user command: '{user_text}'.\n"
        "Identify all Siebel actions mentioned. Respond only with valid JSON.\n"
        "Keys: 'actions' (list). Each action is an object with:\n"
        "  - 'type': one of ['CREATEPRODUCT','CREATEPRICELIST','CREATEPROMOTION','CREATEPRODUCTCLASS',\n"
        "    'CREATEPRODUCTLINE','CREATEPRODUCTATTRIBUTES','CREATEPRODUCTELIGIBILITY','CREATEPRODUCTCOMPATIBILITY']\n"
        "  - 'name': entity name (string, empty if missing)\n"
        "  - For CREATEPRICELIST also include 'product', 'price', 'currency' if provided.\n"
        "Example: {\"actions\": [\n"
        "  {\"type\": \"CREATEPRODUCT\", \"name\": \"iPhone 16\"},\n"
        "  {\"type\": \"CREATEPRICELIST\", \"name\": \"NA pricelist\", \"product\": \"iPhone 16\", \"price\": \"1000\", \"currency\": \"USD\"}\n"
        "]}\n"
        "If no valid action found, respond with {\"actions\": []}."
    )

    response = co.chat(model="command-r", message=prompt)
    try:
        data = json.loads(response.text.strip())
    except json.JSONDecodeError:
        data = {"actions": []}

    print("DEBUG LLM response:", data)
    return {"llm_response": data}

# ----------------------------
# 5. Dispatcher node
# ----------------------------
def dispatcher(state: State) -> State:
    actions = state.get("llm_response", {}).get("actions", [])
    results = []

    if "product_ids" not in state:
        state["product_ids"] = {}

    for action in actions:
        action_type = action.get("type", "").upper()
        name = action.get("name", "")
        if action_type == "CREATEPRICELIST":
            product = action.get("product", "")
            price = action.get("price", "")
            currency = action.get("currency", "")
            result = create_pricelist.invoke({
                "name": name,
                "product": product,
                "price": price,
                "currency": currency,
                "product_ids": state["product_ids"]
            })
        else:
            tool_map = {
                "CREATEPROMOTION": create_promotion,
                "CREATEPRODUCT": create_product,
                "CREATEPRODUCTCLASS": create_product_class,
                "CREATEPRODUCTLINE": create_product_line,
                "CREATEPRODUCTATTRIBUTES": create_product_attributes,
                "CREATEPRODUCTELIGIBILITY": create_product_eligibility,
                "CREATEPRODUCTCOMPATIBILITY": create_product_compatibility,
            }
            tool_fn = tool_map.get(action_type)
            if tool_fn:
                result = tool_fn.invoke({"name": name})
                # If it's a product, store ProductId for later
                if action_type == "CREATEPRODUCT" and isinstance(result, dict):
                    state["product_ids"][name] = result.get("ProductId", "")
            else:
                result = f"Unknown action: {action_type}"

        # Convert dict results to string for joining
        if isinstance(result, dict):
            results.append(json.dumps(result, indent=2))
        else:
            results.append(result)

    if not results:
        results.append(f"You said: {state.get('user_input', '')}")

    return {"result": "\n".join(results)}

# ----------------------------
# 6. Build the LangGraph
# ----------------------------
builder = StateGraph(State)
builder.add_node("interpret", interpret)
builder.add_node("dispatcher", dispatcher)
builder.add_edge(START, "interpret")
builder.add_edge("interpret", "dispatcher")
builder.add_edge("dispatcher", END)
graph = builder.compile()
print(graph)

# ----------------------------
# 7. CLI loop with interactive prompting
# ----------------------------
if __name__ == "__main__":
    print("Siebel Product Model Assistant (type 'exit' or 'quit' to stop)")
    state = {}

    while True:
        user = input("You: ")
        if user.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        state["user_input"] = user

        while True:
            new_state = graph.invoke(state)
            result = new_state.get("result", "…")

            if result.startswith("MISSING_FIELDS:"):
                fields = result.replace("MISSING_FIELDS:", "").split(",")
                for field in fields:
                    value = input(f"Please provide {field}: ")
                    state[field.lower().replace(" ", "_")] = value
                continue  # re-invoke graph with updated state
            else:
                print("Bot:", result)
                break
