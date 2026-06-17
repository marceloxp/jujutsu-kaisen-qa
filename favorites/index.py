from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

def main():
    # Load essay from abramov.txt in Python
    with open("abramov.txt", "r", encoding="utf-8") as file:
        essay = file.read()

    # Settings.embed_model = OllamaEmbedding(model_name="tinyllama:latest")
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # Create Document object with essay
    document = Document(text=essay)

    # Split text and create embeddings. Store them in a VectorStoreIndex
    index = VectorStoreIndex.from_documents([document])

    # Query the index
    query_engine = index.as_query_engine()
    response = query_engine.query({
        "query": "What did the author do in college?",
    })

    # Output response
    print(response.to_string())

if __name__ == "__main__":
    main()
