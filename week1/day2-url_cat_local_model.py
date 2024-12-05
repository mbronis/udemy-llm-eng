import ast
import requests

from bs4 import BeautifulSoup

import ollama


OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"


SYS_PROMPT = """
You are an assistant that analyzes the contents of a website and provides a website categorization.
Based on provided website text, you should return a python dictionary with categories and probability.
Please dont return any other information. Ensure that dict is of te form: {category<str>: probability<float>}.
Return categories that are relevant to the website content, so they can later be used in ml models.
Return categories with probability higher than 10%, or the one with the highest probability, if none is above 10%.
Use a priori categories, does not create new categories based on the website content.

Limit number of categories to 20. Ensure consistency in the categories returned.
Multiple calls with the same webpage should return the same categories.
"""

USER_PROMPT = """
You are looking at a website titled: {title}
The contents of this website is as follows: {text}
I'm interested in page categorization. Return a python dict with categories and probabilities.
Please return just dict without any additional information or comments.
"""


class Website:
    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


def user_prompt_for(website):
    return USER_PROMPT.format(title=website.title, text=website.text)


def messages_for(website):
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt_for(website)},
    ]


def parse_message_to_dict(message):
    try:
        start = (
            0
            if message.find("```python\n") == -1
            else message.find("```python\n") + len("```python\n")
        )
        end = len(message) if message.find("\n```") == -1 else message.find("\n```")
        code_block = message[start:end]
        return ast.literal_eval(code_block)
    except Exception as e:
        print(f"Error parsing message: {e}")
        print(f"Input message:\n{code_block}")
        return None


def categorize(url, model: str = "llama3.2"):
    website = Website(url)
    response = ollama.chat(model=model, messages=messages_for(website))
    raw_content = response["message"]["content"]
    return parse_message_to_dict(raw_content)


SAMPLE_PAGES = [
    "https://news.ycombinator.com",
    "https://www.bbc.co.uk/news",
    "https://weather.com",
    "https://ollama.com",
    "https://www.facebook.com/jarzynova",
    "https://www.allegro.pl",
]


if __name__ == "__main__":
    for url in SAMPLE_PAGES:
        print(f"Categories extracted for {url}:")
        print(categorize(url, model="mistral"))
