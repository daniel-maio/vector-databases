# Quickstart

1.	Set up your API Keys in the .env file
2.	Activate enviroment
3.	Ingest data on Pinecone vector database:
    cmd: python -m documents.ingestion
4.	Enter on your Pinecone account to check the results
5.	Run uvicorn app:app
6.	Copy the http link and past in your browser
7.	Copy a dicionary below (Query Sugestions) and replace in the Edit Value | Schema
8.	Hit the Execute button
9.	The response will be in the Response Body


 
# ---- Query Sugestions --- #

{
"user_id": "id_01",
"message": "How does home office work at the company?"
}


{
"user_id": "id_01",
"message": "I'm new to the company, what do I need to do on my first day?"
}

{
"user_id": "id_01",
"message": "How many vacation days am I entitled to?"
}

{
"user_id": "id_01",
"message": "What documents do I need to submit to HR?"
}

{
"user_id": "id_01",
"message": "How does performance evaluation work?"
}

