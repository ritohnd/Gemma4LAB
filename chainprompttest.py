from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional chef. Answer the user's question briefly."),
    ("user", "{question}")
])

# Initialize the model
llm = ChatOllama(model="gemma4", temperature=0.5)

# Chain the prompt and LLM together
chain = prompt | llm

# Execute the chain
result = chain.invoke({"question": "What can I substitute for eggs in a cake recipe?"})
print(result.content)
