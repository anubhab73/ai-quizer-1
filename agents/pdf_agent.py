# agents/pdf_agent.py
from utils.pdf_utils import load_and_split_pdf
from utils.rag_utils import create_rag_index
import os

def process_pdf(pdf_path: str, pdf_name: str):
    print(f"Processing new PDF: {pdf_name}")
    docs = load_and_split_pdf(pdf_path)
    if not docs:
        print("No text extracted from PDF")
        return None
    vectorstore = create_rag_index(docs, pdf_name)
    return vectorstore