from .interface import LlmModel
from .ollama import OllamaModel
from .openai import OpenAiModel


MODELS = {
    "ollama": OllamaModel,
    "openai": OpenAiModel,
}
