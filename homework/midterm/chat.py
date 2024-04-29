import chainlit as cl
import logging
import sys
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import llama_index
from llama_index.core import set_global_handler

# set_global_handler("wandb", run_args={"project": "meta-10k"})
# wandb_callback = llama_index.core.global_handler

from .globals import (
    DEFAULT_QUESTION1,
    DEFAULT_QUESTION2,
    gpt35_model,
    gpt4_mode
)

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    # Send a response back to the user
    await cl.Message(
        content=f"Received: {message.content}",
    ).send()

@cl.on_chat_start
async def start():

    await cl.Message(
        content="How can I help you about Meta's 2023 10K?"
    ).send()
