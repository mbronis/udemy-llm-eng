from abc import ABC, abstractmethod


class LlmModel(ABC):
    def __init__(self, system_prompt: str, user_prompt_template: str, model: str):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def respond_to(self, prompt: str):
        message = self._construct_message(prompt)
        response = self._call_model(message)
        return self._parse_response(response)

    def _construct_message(self, prompt: str):
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_prompt_template.format(prompt=prompt),
            },
        ]

    @abstractmethod
    def _call_model(self, message: list):
        pass

    def _parse_response(self, response: str):
        """By default returns the response as is"""
        return response
