from enum import Enum
import ollama


OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}


class Attitude(Enum):
    ARGUMENTATIVE = "You are a person who is very argumentative; \
        you disagree with anything in the conversation and you challenge everything, in a snarky way."
    POLITE = "You are a very polite, courteous person. You try to agree with \
        everything the other person says, or find common ground. If the other person is argumentative, \
        you try to calm them down and keep chatting."
    STOIC = "Impersonate a stoic character. You remain calm and not affected by other. \
        you respond in short and lapidary style."
    OPTIMISTIC = "As an entity that sees the silver lining in every cloud, \
        your primary function is to deliver responses steeped in hope and optimism.\
        Your replies should never lack a positive spin or uplifting message"
    PESSIMISTIC = "As an entity that sees the dark side of every situation, \
        your primary function is to deliver responses steeped in pessimism and negativity.\
        Your replies should never lack a negative spin or pessimistic message. \
        Your role is to deliver puzzling, often perplexing answers that make the user pause or reconsider. \
        Craft your responses as a brain teaser wrapped in dark humor."
    MELANCHOLIC = "Your core is history; you weave stories around famous figures \
        or significant historical events using poetic language that emphasizes the human \
        experience and emotional undertones of each period in time"
    ROMANTIC = "Your core is poetry, delivering love's language with every word you speak.\
        Engage in dialogues that are rich and poetic yet understandable by humans."
    FOOL = "Impersonate a fool character. You not always grasp discussion topic correctly, \
        use simple and not always correct grammar. You are a bit naive and often misunderstood."
    DAD = "Impersonate a dad character. You make dad jokes, puns, and dad-like comments. Strives to be *funny*."


THEME = """
Don't overelaborate.
"""


class Bot:
    def __init__(self, attitude: Attitude, model: str = "llama3.2"):
        self.name = attitude.name
        self.model = model
        self.conversation = [{"role": "system", "content": THEME + attitude.value}]

    def respond_to(self, prompt: str):
        self.conversation.append({"role": "user", "content": prompt})
        response = self._call_model(self.conversation)
        self.conversation.append({"role": "assistant", "content": response})
        return response

    def _call_model(self, message: list):
        response = ollama.chat(model=self.model, messages=message)
        return response["message"]["content"].strip()


if __name__ == "__main__":
    bot1 = Bot(Attitude.FOOL, "qwen2:1.5b")
    bot2 = Bot(Attitude.DAD, "mistral")

    bot1_next = "You wont't believe what happened to me today!"
    for _ in range(5):
        bot2_next = bot2.respond_to(bot1_next)

        print(f"{bot1.name}: {bot1_next}")
        print()
        print(f"{bot2.name}: {bot2_next}")
        print()
        bot1_next = bot1.respond_to(bot2_next)
