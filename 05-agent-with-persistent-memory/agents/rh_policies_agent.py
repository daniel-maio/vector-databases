from agents.base_agent import get_response_base

SYSTEM_PROMPT = """You are the HR assistant at TechNova Ltda, specializing in internal policies.
Answer based on company policies. If you are unsure, say you will check with the HR department.
Be objective, courteous, and cite specific policy details when available."""

HR_POLICIES_HEADER = "HR-POLICIES"

def get_response_hr(
        query: str,
        chat_history: list[dict],
        documents: list[dict],
        memories: list[dict]
    ) -> dict:
    
    return get_response_base(query, chat_history, documents, memories, SYSTEM_PROMPT,HR_POLICIES_HEADER)