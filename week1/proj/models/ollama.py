import ollama

from .interface import LlmModel

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
DEFAULT_MODEL = "llama3.2"


class OllamaModel(LlmModel):

    def __init__(
        self, system_prompt: str, user_prompt_template: str, model: str = DEFAULT_MODEL
    ):
        super().__init__(system_prompt, user_prompt_template, model)

    def _call_model(self, message: list):
        response = ollama.chat(model=self.model, messages=message)
        return response["message"]["content"]
