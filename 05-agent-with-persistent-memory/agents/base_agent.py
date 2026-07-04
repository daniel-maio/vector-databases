from client import groq_client
from documents.format_doc import format_doc
from typing import Optional


def _build_context(
        retrieved_documents: Optional[list[dict]],
        retrieved_memories: Optional[list[dict]],
        document_header: str) -> str:
    
    context = []

    if retrieved_documents:

        context.append(f"=== {document_header} ===")
        
        for doc in retrieved_documents:             
    
            context.append(f"{doc['title']}:\n{doc['content']}")

    if retrieved_memories:

        context.append(f"=== User Previous Conversations ===")

        for memory in retrieved_memories:             
            
            context.append(f"- {memory['summary']}")

    return "\n\n".join(context)


def get_response_base(
        query: str,
        chat_history: list[dict],
        retrieved_documents: list[dict],
        retrieved_memories: list[dict],
        system_prompt: str,
        document_header: str,
        relevance_threshold=0.5) -> dict:
    
    """Generates a response from an agent using LLM with RAG context and memories.

    This function is called by both the policy agent and the onboarding agent.

    The difference between them comes from the system_prompt and header_documents parameters.

    Arg:

    query: The user's current question.

    chat_history: Previous messages in the conversation (for dialogue continuity).

    retrieved_documents: Relevant RAG documents retrieved from Pinecone.

    retrieved_memories: User's long-term memories.

    system_prompt: Agent-specific behavior instructions.

    header_documents: Header for the documents section in the context.

    relevance_threshold: a Threshold for determining wether the interaction should be stored as memory.

    Returns:

    Dict with 'response' (generated text) and 'memorize' (bool indicating whether the interaction should be saved as memory).
    """

    context = _build_context(retrieved_documents, retrieved_memories, document_header)

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    if context:
        messages.append(
            {"role": "system", "content": f"Available context:\n{context}"}
        )
    
    if chat_history:
        for msg in chat_history[-10:]:
            messages.append(
                {"role":msg.get("role"), "content": msg.get("content")}
            )
 
    messages.append({"role": "user", "content": query})

    chat_completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    response = chat_completion.choices[0].message.content.strip()

    memorize = any(doc.get("score", 0) >= relevance_threshold for doc in retrieved_documents)

    return {"response": response, "memorize": memorize}



