import models as m


SYS_PROMPT = """
You are an assistant that explains technical questions about python code snippets.
Based on provided code snippet, you should return with structured response in markdown.
First explain what the code does. Then propose code improvements or suggestions if needed.
Finally provide example use cases with actual input and expected output in the form of python code block.
"""

USER_PROMPT_TEMPLATE = """
Help me understand what the following code does:
{prompt}
"""


CODE_SNIPPETS = [
    "print('Hello, Adam!')",
    "return yield {book.author for book in books if book.get('author')}",
]

if __name__ == "__main__":
    model = m.OllamaModel(SYS_PROMPT, USER_PROMPT_TEMPLATE)
    print(model.respond_to(CODE_SNIPPETS[-1]))

    model = m.OllamaModel(SYS_PROMPT, USER_PROMPT_TEMPLATE, model="mistral")
    print(model.respond_to(CODE_SNIPPETS[-1]))

    model = m.OpenAiModel(SYS_PROMPT, USER_PROMPT_TEMPLATE)
    print(model.respond_to(CODE_SNIPPETS[0]))
