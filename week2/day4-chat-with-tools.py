import json

import gradio as gr
import ollama


MODEL = "llama3.2"


system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."
system_message += "Ask for departure city if needed."


ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}


def get_ticket_price(destination_city: str):
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")


price_function = {
    "name": "get_ticket_price",
    "description": """
        Get the price of a return ticket to the destination city.
        Call this whenever you need to know the ticket price,
        for example when a customer asks 'How much is a ticket to this city'
        """,
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}
tools = [{"type": "function", "function": price_function}]
# We have to write that function handle_tool_call:


def handle_tool_call(tool_call):
    arguments = tool_call["function"]["arguments"]
    city = arguments.get("destination_city")
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city, "price": price}),
    }
    print(f"Tool response: {response}")
    return response


def chat(message, history):
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
        if msg.get("role") in ("user", "assistant")
    ]
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )

    response = ollama.chat(model=MODEL, messages=messages, tools=tools)
    print("---------------------------")
    print("response with tool call", response)
    tool_call = response["message"].get("tool_calls")
    if tool_call:
        tool_response = handle_tool_call(tool_call[0])
        messages.append(tool_response)
        print("---------------------------")
        print("message with tool response:", messages)
        response = ollama.chat(model=MODEL, messages=messages)
        print("---------------------------")
        print("response to message with tool resp added:", response)

    return response["message"]["content"].strip()


if __name__ == "__main__":
    gr.ChatInterface(fn=chat, type="messages").launch()
