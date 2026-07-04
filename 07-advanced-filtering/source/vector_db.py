# http://localhost:6333/dashboard

import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, Range

class VectorDB():
    def __init__(self):
        self.client = QdrantClient(url=os.getenv('QDRANT_URL'))
        self.collection_name = os.getenv('COLLECTION_NAME')

    # Cria collection
    def create_collection(self):
        '''
        Cria Collection e os Índices (Payload Schema).
        '''

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        print(f"Collections: {self.client.get_collections()}")

        # payload schema
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="category",
            field_schema="keyword"
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="brand",
            field_schema="keyword"
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="price",
            field_schema="integer"
        )
    
    def upload_data(self, products, embedding_model):
        
        '''Insere os vetores e o payload no banco vetorial.
        '''
        points = []
        for i, item in enumerate(products):
            text_to_embed = f"{item['name']} - {item['description']}."

            vector = embedding_model.get_embedding(text_to_embed)

            points.append(PointStruct(
                id = i,
                vector=vector,
                payload = item # o que foi marcado em create_payload_index, ele vai puxar para indexar.
                )
            )
    
        self.client.upsert(collection_name=self.collection_name, points = points)
        
        print(f"{len(points)} produtos inseridos.")

    def search(self, vector_query, category=None, brand = None, min_price = None, max_price = None, limit=2):

        '''A função começa inicializando uma lista vazia de condições de filtro. Essa lista será
        usada para montar, de forma incremental, as restrições que devem ser aplicadas
        junto à busca vetorial, dependendo apenas dos parâmetros que o usuário realmente
        informou.'''
        filter_conditions = []
         
        '''A condição criada usa FieldCondition com MatchValue,
        o que significa uma correspondência exata no campo category,
        aproveitando o índice de keyword criado na coleção'''
        if category != None:
            filter_conditions.append(FieldCondition(key = 'category', match = MatchValue(value=category)))
        
        if brand != None:
            filter_conditions.append(FieldCondition(key = 'brand', match = MatchValue(value=brand)))

        if min_price is not None or max_price is not None:
            filter_conditions.append(FieldCondition(key = 'price', range = Range(gte=min_price, lte=max_price)))

        #print(filter_conditions)

        filter = Filter(must=filter_conditions) if filter_conditions else None

        '''Na sequência, é executada a consulta vetorial propriamente dita. O método
        query_points recebe o nome da coleção, o vetor de consulta gerado pelo modelo de
        embeddings, o filtro opcional e o limite de resultados. O Qdrant então combina a
        busca por similaridade semântica com as restrições de payload de forma eficiente,
        retornando apenas os pontos que satisfazem ambos os critérios.'''
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query = vector_query,
            query_filter=filter,
            limit=limit
        )
        return results.points


# Como os points retornam:

# [
# ScoredPoint(id=2, version=7, score=0.4677721, payload={'name': 'MacBook Air M2', 'description': 'Notebook leve, bateria de longa duração e chip M2 ultra rápido.', 'category': 'Notebooks', 'brand': 'Apple', 'price': 8000}, vector=None, shard_key=None, order_value=None),
# ScoredPoint(id=3, version=7, score=0.27871495, payload={'name': 'Dell XPS 13', 'description': 'Notebook Windows premium com tela infinita e alta portabilidade.', 'category': 'Notebooks', 'brand': 'Dell', 'price': 7500}, vector=None, shard_key=None, order_value=None)
# ]