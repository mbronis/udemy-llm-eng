import os
import requests
import ast
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI


SYS_PROMPT = """
You are an assistant that analyzes the contents of a website and provides a website categorization.
Based on provided website text, you should return a python dictionary with categories and probability.
"""

USER_PROMPT = """
You are looking at a website titled: {title}
The contents of this website is as follows: {text}
I'm interested in page categorization. Return a python dict with categories and probabilities.
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
        start = message.find("```python\n") + len("```python\n")
        end = message.find("\n```")
        code_block = message[start:end]

        parsed_dict = ast.literal_eval(code_block)
        return parsed_dict["categories"]
    except Exception as e:
        print(f"Error parsing message: {e}")
        return None


def categorize(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages_for(website)
    )
    raw_content = response.choices[0].message.content
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
    # get the API key and init the OpenAI object
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Check the key
    if not api_key:
        print(
            "No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!"
        )
    elif not api_key.startswith("sk-proj-"):
        print(
            "An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook"
        )
    elif api_key.strip() != api_key:
        print(
            "An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook"
        )
    else:
        print("API key found and looks good so far!")

    openai = OpenAI()

    for url in SAMPLE_PAGES:
        print(f"Categories extracted for {url}:")
        print(categorize(url))
