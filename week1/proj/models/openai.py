import os
from dotenv import load_dotenv

from openai import OpenAI

from .interface import LlmModel


DEFAULT_MODEL = "gpt-4o-mini"


class OpenAiModel(LlmModel):

    def __init__(
        self, system_prompt: str, user_prompt_template: str, model: str = DEFAULT_MODEL
    ):
        super().__init__(system_prompt, user_prompt_template, model)
        load_dotenv()
        _ = os.getenv("OPENAI_API_KEY")
        self.chat = OpenAI()

    def _call_model(self, message: list):
        response = self.chat.chat.completions.create(model=self.model, messages=message)
        return response.choices[0].message.content
