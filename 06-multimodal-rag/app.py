import os
import shutil
import uuid
from typing import TypedDict, Dict, List, Any, Optional

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager

from create_db import generate_embedding, get_client


### Variáveis ###

COLLECTION_NAME = 'catalog'

llm = ChatOllama(model = "llama3", temperature = 0.7)

#############

class AgentState(TypedDict):
    image_path: str
    results: Optional[List[Dict[str, Any]]]
    response: Optional[str]


def search_node(state: AgentState):

    query_emb = generate_embedding(state["image_path"])

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_emb],
        anns_field='vector',
        search_params={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}},
        limit=3,
        output_fields=['name', 'price', 'description'],
    )

    matches = [
        {
            'name': hit.entity.get('name'),
            'price': hit.entity.get('price'),
            'description': hit.entity.get('description'),
            'cosine': round(hit.distance, 4)
        }
        for hit in results[0]
        if hit.distance >= 0.5
    ]

    return {'results': matches}

def agent_node(state: AgentState):
    
    products = state.get('results', [])

    if not products:
        prompt = f"Nenhum produto similar encontrado. Pergunte se o cliente deseja outro item."
        response = llm.invoke(prompt)
        return {'response': response.content}
    
    else:
        text = '\n'.join([
            f"Nome: {p['name']} | Preço: R$ {p['price']} | Descrição: {p['description']}."
            for p in products
        ])

        prompt = f"""
        Você é um assistente de vendas de moda.
        O cliente enviou uma foto e encontramos estes produtos similares no nosso catálogo visual:
        {text}

        1. Confirme que temos produtos parecidos com o da foto.
        2. Descreva brevemente a melhor opção.
        3. Use um tom persuasivo para fechar a venda.
        4. Se houver mais de uma opção, ofereça como alternativa.
        Responda em Português do Brasil.
        """

        response = llm.invoke(prompt)

        return {'response': response.content}
    
# Orquestrador do pipeline usando LangGraph
def build_graph():
    workflow = StateGraph(state_schema=AgentState)

    workflow.add_node('search_node', search_node)
    workflow.add_node('agent_node', agent_node)

    workflow.set_entry_point('search_node')
    
    workflow.add_edge('search_node', 'agent_node')
    workflow.add_edge('agent_node', END)

    return workflow.compile()

agent = build_graph()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global client
    client = get_client()
    client.load_collection(COLLECTION_NAME)
    yield
    # Shutdown
    client.close()

app = FastAPI(
    title = "Visual Search Agent API",
    description="Busca visual de produtos com Milvus + LLM",
    lifespan = lifespan)

@app.get("/status")
def check_client():
    if client is None:
        return {"error": "client is None"}
    return {"status": "ok"}

@app.post('/search')
async def search(file: UploadFile = File(...)):
    
    file_id = str(uuid.uuid4())
    
    upload_dir = "static/uploads"
    file_path = f"{upload_dir}/{file_id}.jpg"
    
    os.makedirs(upload_dir, exist_ok=True)
    
    try:
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        initial_state = {
            'image_path': file_path,
            'results': None,
            'response': None
        }

        result = agent.invoke(initial_state)

        return {
        'products': result.get('results', []),
        'response': result.get('response', "")
        }
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)