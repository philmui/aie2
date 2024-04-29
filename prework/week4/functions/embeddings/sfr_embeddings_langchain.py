from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
import requests
import json


class SFREmbeddingForLangChain(BaseModel, Embeddings):
    address: str = 'https://rag.salesforceresearch.ai'
    model_id: str = 'sfr_embedding_mistral'
    """
    Ref: https://api.python.langchain.com/en/latest/_modules/langchain_community/embeddings/huggingface.html#HuggingFaceEmbeddings
    """
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts, is_query=False)

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text, is_query=True)

    def embed(self, text: str | list[str], is_query: bool = True) -> list[float] | list[list[float]]:
        content = {
            'text': text,
            'model_id': self.model_id,
            'is_query': str(is_query),
        }
        response = requests.post(
            url=f'{self.address}/embedding',
            json=content,
        )
        output = self.parse_response(response)
        return output

    @classmethod
    def parse_response(cls, response):
        return json.loads(response.content)['embedding']
    

def main():
    from pprint import pprint

    query = "Find the relevant regulation or policy that addresses a customer's right to deposit funds without " \
            "unnecessary inconvenience or red tape, in the context of a bank's branch experience, and how to handle " \
            "cases of rude staff."
    document = "This is a sample document."
    documents = [document] * 2

    embedding_engine = SFREmbeddingForLangChain()
    query_embedding = embedding_engine.embed_query(query)
    document_embeddings = embedding_engine.embed_documents(documents)

    for output in (query_embedding, document_embeddings):
        pprint(output)
        print('=' * 16)


if __name__ == '__main__':
    main()
