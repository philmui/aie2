import os
import chainlit as cl
import asyncio

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    HumanMessage, 
    SystemMessage
)

MESSAGE_HISTORY = "message_history"
GPT3_MODEL_NAME = "gpt-3.5-turbo-0125"
GPT4_MODEL_NAME = "gpt-4-turbo-preview"
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

_system_message = SystemMessage(content=(
    "You are a helpful and creative assistant who tries your best to answer questions "
    " in the most witty and funny way.  If you feel that a poetic touch is neede to "
    " uplift the mood for the user, go ahead to write a sonnet.  Always be positive, "
    " encouraging, and inspirational if possible."
))
_chat_model: ChatOpenAI

@cl.on_chat_start
async def start():
    global _chat_model
    _chat_model = ChatOpenAI(
        model=GPT4_MODEL_NAME, 
        temperature=0.5
    )
    cl.user_session.set(MESSAGE_HISTORY, [_system_message])
    await cl.Message(
        content="Hello there!  How are you?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
        
    messages = cl.user_session.get(MESSAGE_HISTORY) 
    if not messages:
        messages = [_system_message]

    if len(message.elements) > 0:
        for element in message.elements:
            with open(element.path, "r") as uploaded_file:
                content = uploaded_file.read()
            messages.append(HumanMessage(content=content))
            confirm_message = cl.Message(content=f"Uploaded file: {element.name}")
            await confirm_message.send()

    messages.append(HumanMessage(content=message.content))
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    print(vars(prompt))
    output_parser = StrOutputParser()

    try:
        chain = prompt | _chat_model | output_parser
        response = chain.invoke({"input": message.content})
    except Exception as e:
        response = f"no response: {e}"

    await cl.Message(
        content=response
    ).send()
