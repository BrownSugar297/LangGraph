from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

generation_llm = ChatGroq(
    model="llama-3.3-70b-versatile",   
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

reflection_llm = ChatGroq(
    model="llama-3.1-8b-instant",      
    temperature=0.5,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts. "
            "Generate the best twitter post possible for the user's request. "
            "If the user provides critique, respond with a revised version.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. "
            "Give detailed critique and improvement suggestions.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_chain = generation_prompt | generation_llm
reflection_chain = reflection_prompt | reflection_llm