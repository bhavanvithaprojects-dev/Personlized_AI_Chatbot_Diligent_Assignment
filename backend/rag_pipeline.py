from backend.vector_db import retrieve_context
from backend.llm import generate_response

def get_answer(query: str):
    context = retrieve_context(query)

    prompt = f"""
You are Jarvis, a helpful AI assistant.

Context:
{context}

User Question:
{query}

Answer clearly:
"""
    return generate_response(prompt)
