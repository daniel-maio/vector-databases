import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLMService:
    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Constrói o texto de contexto a partir dos produtos retornados pelo banco vetorial que será passado ao LLM.
    def generate_recommendation(self, user_query, products_retrieved):
        
        payloads = [point.payload for point in products_retrieved]

        context_str = "\n---\n".join([
            f"Name: {item['name']}, category: {item['category']}, brand: {item['brand']}, price: {item['price']}, description: {item['description']}"
            for item in payloads
            ]
        )

        # Constrói os prompts

        system_prompt = f"""
        Você é um assistente de vendas especialista em produtos eletrônicos.
        PRODUTOS DISPONÍVEIS: {context_str}.
        INSTRUÇÃO: Com base apenas nos produtos acima, recomende a melhor opção para o cliente.
        Explique por que o produto atende à necessidade dele. Seja persuasivo, mas honesto.
        Se nenhum produto parecer adequado, diga isso educadamente.

        Sua resposta:
        """

        user_prompt = f"""PERGUNTA DO CLIENTE: {user_query}."""

        # Realiza a chamada ao modelo de LLM para gerar a resposta
        chat_completion = self.client.chat.completions.create(
            messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content

        
