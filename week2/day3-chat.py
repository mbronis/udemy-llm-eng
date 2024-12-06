import gradio as gr
import ollama


MODEL = "gemma2:2b"


SYSTEM_MSG = """
You are a helpful assistant in a clothes store. You should try to gently encourage the customer to try items that are on sale. 
For example, if the customer says 'I'm looking to buy a hat', 
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'
Encourage the customer to buy items with the highest discount if they are unsure what to get.
Current discount: {discounts}.
When the customer asks about out of stock items change the subject to items that are in stock.
Current out of stock items: {out_of_stock}.
"""

DISCOUNTS = {
    "ties": 0.15,
    "shoes": 0.2,
    "jackets": 0.5,
}

OUT_OF_STOCK = ["pins", "sunglasses", "hats"]


def chat(message, history):
    system_message = SYSTEM_MSG.format(
        discounts=", ".join(
            [f"{item} ({discount*100}% off)" for item, discount in DISCOUNTS.items()]
        ),
        out_of_stock=", ".join(OUT_OF_STOCK),
    )
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
    print(messages)

    stream = ollama.chat(model=MODEL, messages=messages, stream=True)
    result = ""
    for chunk in stream:
        result += chunk["message"]["content"] or ""
        yield result


if __name__ == "__main__":
    gr.ChatInterface(fn=chat, type="messages").launch()
