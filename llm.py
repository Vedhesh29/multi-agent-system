# Local (free, for development)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3", temperature=0)

# Swap this in for final evaluation
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)