import sys
import json
from source.embeddings import EmbeddingModel
from source.vector_db import VectorDB
from source.llm import LLMService

print(f"\nPython Executável: {sys.executable}")

def ingestao_dados():

    print(f"Iniciando Ingestão de Dados...")

    # Carrega o arquivo que contém os produtos
    with open('data/produtos.json','r', encoding='utf-8') as file:
        produtos = json.load(file)
    
    # Inicializa o modelo de Embeddings
    embed_model = EmbeddingModel()

    # Inicializa o Banco Vetorial
    vector_db = VectorDB()

    #Cria a Collection e os índices (payload schema)
    vector_db.create_collection()

    # Insere os points
    vector_db.upload_data(produtos, embed_model)
    
    print(f"Ingestão Concluída.")
    return vector_db, embed_model

# Preciso de um notebook bom para viagens.
# Qual o melhor headphone com cancelamento de ruído?
# Qual console de games suporta 4k?
# Qual o melhor automóvel com baixo consumo de combustível?

###############


# 1 - Abrir o Docker e Rodar o container

# 2 - Ativar o env

# 3 - no cmd: python main.py

vector_db, embed_model = ingestao_dados()

user_query = 'Qual console de games suporta 4k?'

vector_query = embed_model.get_embedding(user_query)

llm_service = LLMService()

points = vector_db.search(vector_query)

response = llm_service.generate_recommendation(user_query, points)

print(f"\nResposta do Assistente Virtual: {response}")

print(f"Collections: {vector_db.client.get_collections()}")