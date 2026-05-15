# utils/rag_utils.py
import os
import time

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-quizer")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "current-pdf")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


def _get_pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing. Add it to your .env file.")

    pc = Pinecone(api_key=api_key)
    indexes = pc.list_indexes()
    if hasattr(indexes, "names"):
        index_names = indexes.names()
    else:
        index_names = []
        for index in indexes:
            if isinstance(index, dict):
                index_names.append(index["name"])
            else:
                index_names.append(index.name)

    if PINECONE_INDEX_NAME not in index_names:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

        while not _is_index_ready(pc):
            time.sleep(1)

    return pc.Index(PINECONE_INDEX_NAME)


def _is_index_ready(pc):
    status = pc.describe_index(PINECONE_INDEX_NAME).status
    if isinstance(status, dict):
        return status.get("ready", False)
    return getattr(status, "ready", False)


def _clear_namespace(index):
    try:
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        print(f"Cleared Pinecone namespace: {PINECONE_NAMESPACE}")
    except Exception as exc:
        print(f"Could not clear Pinecone namespace '{PINECONE_NAMESPACE}': {exc}")


def create_rag_index(docs, pdf_name: str = "current_pdf"):
    """
    Always creates/overwrites the Pinecone namespace used by the current PDF.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index = _get_pinecone_index()
    _clear_namespace(index)

    print(f"Creating Pinecone vector store for: {pdf_name}")
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE,
    )
    print("Pinecone vector store created")
    return vectorstore


def retrieve_context(vectorstore, query: str, k: int = 4):
    if not vectorstore:
        return ""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as exc:
        print(f"Pinecone retrieval error: {exc}")
        return ""
