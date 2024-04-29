from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

GPT4_MODEL_NAME = "gpt-4-turbo-2024-04-09"
GPT35_MODEL_NAME = "gpt-3.5-turbo-1106"

gpt35_model = OpenAI(model=GPT35_MODEL_NAME, temperature=0.0)
gpt4_mode   = OpenAI(model=GPT4_MODEL_NAME, temperature=0.0)

Settings.llm = OpenAI(model=gpt35_model)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

DEFAULT_QUESTION1 = "What was the total value of 'Cash and cash equivalents' as of December 31, 2023?"
DEFAULT_QUESTION2 = "Who are Meta's 'Directors' (i.e., members of the Board of Directors)?"