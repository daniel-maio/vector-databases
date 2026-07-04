# --- Imports --- #
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from operator import itemgetter

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

pdf_path = "data/hr_policies_vacation_guidelines.pdf"

# --- Program --- #

class RAGEngine():
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

        self.llm_model = ChatGroq(model = "llama-3.3-70b-versatile",
                                  temperature=0.0,
                                  api_key=GROQ_API_KEY)

        self.vector_store = Chroma(
            collection_name = 'rh_policies',
            embedding_function = self.embedding_model,
            persist_directory = "./chroma_db"
        )

        self.retriever = self.vector_store.as_retriever(
            search_type = 'similarity',
            search_kwargs = {'k':2})

        self.retrieved_docs = None
        
    def ingest_documents(self, pdf_path):

        loader = PDFPlumberLoader(pdf_path)
        docs = loader.load()

        if not docs:
            return f"PDF not found."

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 400,
                                                       chunk_overlap = 100)

        split_docs = text_splitter.split_documents(docs)

        self.vector_store.add_documents(documents = split_docs)

        print(f"{len(split_docs)} chunks added to Collection.")


    def format_docs(self, docs):
        return "\n\n".join(
            [doc.page_content for doc in docs]
        )
    
    def retrieve_source(self, query):

        docs = self.retriever.invoke(query)
        
        return "\n\n".join([f"Source: {doc.metadata['source']}\nPage: {doc.metadata['page']}"for doc in docs])
    
    
    def get_response(self, query):

        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                You are a precise assistant.

                Use ONLY the context below to answer.
                If the answer is not contained in the context, say: "I don't know based on the provided context."

                Context:
                {context}
                """
                ),
                (
                    "user", "{question}"
                )
            ]
        )
        
        rag_chain = (
            {
                "context": itemgetter("question") | self.retriever | self.format_docs,
                "question": itemgetter("question")
            }
            | template
            | self.llm_model
            | StrOutputParser()
        )

        return rag_chain.invoke(
            input = {"question": query})

    
    def clear_database(self):

        self.vector_store.delete_collection()

        self.vector_store = Chroma(
            collection_name = 'rh_policies',
            embedding_function = self.embedding_model,
            persist_directory = "./chroma_db"
        )

        return f"Database deleted."


rag_engine = RAGEngine()

#rag_engine.clear_database()

rag_engine.ingest_documents(pdf_path)

chunks = rag_engine.vector_store._collection.get()

query = "What happens if I need to take a few sick days in a row?"

source = rag_engine.retrieve_source(query)

answer = rag_engine.get_response(query)

print(f"{source}")
print('-' * 30)
print(f'Response: {answer}')

