import openai
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class ChatOpenAI:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")

    def run(self, messages, text_only: bool = True, **kwargs):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        openai.api_key = self.openai_api_key
        response = openai.Client().chat.completions.create(
            model=self.model_name, messages=messages,
            **kwargs,
        )

        if text_only:
            return response.choices[0].message.content

        return response
    
if __name__ == "__main__":
    print("Testing ChatOpenAI")
    chat_openai = ChatOpenAI()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life?"}
    ]
    response = chat_openai.run(messages)
    print(response)
