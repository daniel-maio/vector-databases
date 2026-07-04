from agents.base_agent import get_response_base

SYSTEM_PROMPT = """You are the onboarding assistant for TechNova Ltda.
Help new employees with all their questions about their first days at the company.
Be welcoming and detail-oriented.
Use the information in the onboarding guide to give accurate and complete answers."""

ONBOARDING_HEADER = "ONBOARDING"

def get_response_onboard(
        query: str,
        chat_history: list[dict],
        documents: list[dict],
        memories: list[dict]
    ) -> dict:
    
    return get_response_base(query, chat_history, documents, memories, SYSTEM_PROMPT, ONBOARDING_HEADER)