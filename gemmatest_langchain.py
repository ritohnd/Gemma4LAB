from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Initialize the Gemma 4 model
llm = ChatOllama(
    model="gemma4:e4b",
    temperature=0.7,
)

# Invoke the model
messages = [
    HumanMessage(content="What model are you ? Like the name and specs, Like I know you are Gemma 4 but which one ? 26b, e2b what ?")
]

response = llm.invoke(messages)
print(response.content)
