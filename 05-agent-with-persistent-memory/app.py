# --- Imports --- #

from pydantic import BaseModel

from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from agents.graph import graph

# --- APP --- #

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    agent_type: str
    memories: list[dict]
    documents: list[dict]
    chat_history: list[dict]

app = FastAPI()

@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="TechNova API"
    )

@app.post("/chat", response_model=ChatResponse)
def initialize(request: ChatRequest):

    initial_state = {
        "user_id": request.user_id,
        "last_message": request.message
    }           

    result = graph.invoke(
        input=initial_state,
        config={"configurable": {"thread_id": "123"}})

    return ChatResponse(
        response=result["response"],
        agent_type=result["agent_type"],
        memories=result["memories"],
        documents=result["documents"],
        chat_history=result['chat_history'][-6:]
    )


