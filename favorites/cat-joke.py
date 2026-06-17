from langchain_community.llms import Ollama

print("Carregando modelo...")
llm = Ollama(
    model="gemma:2b"
)  # assuming you have Ollama installed and have llama3 model pulled with `ollama pull llama3 `

print("Efetuando consulta...")
result = llm.invoke("Tell me a cat joke")

print(result)
