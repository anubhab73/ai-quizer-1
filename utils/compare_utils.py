# utils/compare_utils.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Shared embedder
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

def embedding_similarity(text1: str, text2: str) -> float:
    if not text1.strip() or not text2.strip():
        return 0.0
    emb1 = get_embedder().embed_query(text1)
    emb2 = get_embedder().embed_query(text2)
    sim = cosine_similarity([emb1], [emb2])[0][0]
    return max(0.0, min(1.0, float(sim)))

# LLM for feedback only
_llm_chain = None
def get_feedback_chain():
    global _llm_chain
    if _llm_chain is None:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
        prompt = PromptTemplate(
            input_variables=["model_answer", "student_answer", "similarity"],
            template="""
            Model Answer: {model_answer}
            Student Answer: {student_answer}
            Semantic Similarity: {similarity:.2%}

            Based on the similarity score and content, provide:
            - Constructive feedback (2–4 sentences)
            - 2–4 specific weak areas (if any)

            Return ONLY JSON:
            {{"feedback": "Your answer...", "weak_areas": ["Missed X", "Unclear on Y"]}}
            """
        )
        parser = JsonOutputParser()
        fixer = OutputFixingParser.from_llm(parser=parser, llm=llm)
        _llm_chain = prompt | llm | fixer
    return _llm_chain

def hybrid_evaluate(model_answer: str, student_answer: str):
    if not student_answer or not student_answer.strip():
        return {
            "score": 0.0,
            "feedback": "No answer provided.",
            "weak_areas": ["No response submitted"]
        }

    # 1. Get objective similarity score
    similarity = embedding_similarity(model_answer, student_answer)
    base_score = round(similarity * 10, 1)

    # 2. Use LLM only for rich feedback
    try:
        chain = get_feedback_chain()
        fb = chain.invoke({
            "model_answer": model_answer,
            "student_answer": student_answer,
            "similarity": similarity
        })
        feedback = fb.get("feedback", "Good effort.")
        weak_areas = fb.get("weak_areas", [])
    except:
        feedback = "Your answer shows some understanding."
        weak_areas = ["Review key concepts"] if similarity < 0.7 else []

    return {
        "score": base_score,
        "feedback": feedback,
        "weak_areas": weak_areas
    }