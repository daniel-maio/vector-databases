Agente de busca visual para e-commerce que recebe uma imagem, encontra produtos similares usando embeddings (CLIP) e Milvus, e gera respostas de vendas com LLM via LangGraph e FastAPI.

# Visual Search Agent API

## Quickstart

1. Suba o banco vetorial:\
    • docker-compose up -d

2. Crie o ambiente, instale as dependências e ative o ambiente:\
    • conda create --name nome_do_env\
    • pip install -r requirements.txt\
    • conda activate nome_do_env

3. Crie a base de dados e a collection:\
    • python create_db.py

4. Inicie a API:\
    • uvicorn api:app --port 8000

5. Acesse o endereço: localhost:8000/docs

6. Clique em Search → Try It out → Escolher Arquivo (escolha a imagem) → Execute.

7. Aguardar a resposta do LLM.



