import requests
from dotenv import load_dotenv
from enum import Enum

import gradio as gr
from bs4 import BeautifulSoup

import ollama
from openai import OpenAI


# OLLAMA_API = "http://localhost:11434/api/chat"
# HEADERS = {"Content-Type": "application/json"}


class Tone(Enum):
    Formal = "You are an assistant that analyzes the contents of a company website landing page \
        and creates a short brochure about the company for prospective customers, investors and recruits. \
        Respond in markdown."
    Informal = "You are an assistant that analyzes the contents of a company website landing page \
        and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits.\
        Include details of company culture, customers and careers/jobs if you have the information. \
        Respond in markdown."
    Short = "You are an assistant that analyzes the contents of a company website landing page \
        and creates a very short synopsis about the company for prospective customers, investors and recruits.\
        Keep result as short as possible. Respond in markdown."


class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        self.body = response.content
        soup = BeautifulSoup(self.body, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


def message_gpt(prompt):
    openai = OpenAI()
    messages = [
        {"role": "system", "content": Tone.Informal.value},
        {"role": "user", "content": prompt},
    ]
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return completion.choices[0].message.content


def regular_ollama(prompt, tone: str):
    MODEL = "gemma2:2b"
    messages = [
        {"role": "system", "content": tone},
        {"role": "user", "content": prompt},
    ]
    result = ollama.chat(model=MODEL, messages=messages)
    return result["message"]["content"].strip()


def regular_ollama2(prompt):
    MODEL = "gemma2:2b"
    messages = [
        {"role": "system", "content": Tone.Informal.value},
        {"role": "user", "content": prompt},
    ]
    result_raw = ollama.chat(model=MODEL, messages=messages)
    result = result_raw["message"]["content"].strip()
    print(result)
    return result


def stream_gpt(prompt, tone: str):
    openai = OpenAI()
    messages = [
        {"role": "system", "content": tone},
        {"role": "user", "content": prompt},
    ]
    stream = openai.chat.completions.create(
        model="gpt-4o-mini", messages=messages, stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


def stream_ollama(prompt, tone: str):
    MODEL = "gemma2:2b"
    messages = [
        {"role": "system", "content": tone},
        {"role": "user", "content": prompt},
    ]
    stream = ollama.chat(model=MODEL, messages=messages, stream=True)
    result = ""
    for chunk in stream:
        result += chunk["message"]["content"] or ""
        yield result


def stream_brochure(url, model, tone_name):
    prompt = f"Please generate a company brochure for a company in Markdown. Here is their landing page."
    prompt += "Infere company's name from the title or page content.\n"
    prompt += Website(url).get_contents()
    print(prompt)

    if model == "GPT":
        result = stream_gpt(prompt, Tone[tone_name].value)
    elif model == "OLLAMA":
        result = stream_ollama(prompt, Tone[tone_name].value)
    else:
        raise ValueError("Unknown model")
    print(result)
    yield from result


if __name__ == "__main__":
    load_dotenv()

    view = gr.Interface(
        fn=stream_brochure,
        inputs=[
            gr.Textbox(label="Landing page URL including http:// or https://"),
            gr.Dropdown(["GPT", "OLLAMA"], label="Select model"),
            gr.Dropdown([t.name for t in Tone], label="Select tone"),
        ],
        outputs=[gr.Markdown(label="Brochure:")],
        flagging_mode="never",
    )
    view.launch()
