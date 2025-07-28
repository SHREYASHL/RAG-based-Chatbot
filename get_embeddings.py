from langchain_community.embeddings import OllamaEmbeddings


def get_embeddings():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings