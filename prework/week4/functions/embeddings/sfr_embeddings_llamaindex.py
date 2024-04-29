from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
import requests
import json
from llama_index.core.callbacks.global_handlers import set_global_handler

set_global_handler(eval_mode="simple")


class SFREmbeddingsForLlamaIndex(BaseEmbedding):
    address: str = 'https://rag.salesforceresearch.ai'
    model_id: str = 'sfr_embedding_mistral'
    """
    Ref: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-gemini/llama_index/embeddings/gemini/base.py
    """
    def _get_query_embedding(self, query: str) -> Embedding:
        return self.embed(query, is_query=True)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self.embed(text, is_query=False)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self.embed(texts, is_query=False)

    def embed(self, text: str | list[str], is_query: bool = True) -> Embedding | list[Embedding]:
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

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._get_text_embeddings(texts)

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

    embedding_engine = SFREmbeddingsForLlamaIndex()
    query_embedding = embedding_engine.get_query_embedding(query)
    document_embedding = embedding_engine.get_text_embedding(document)
    document_embeddings = embedding_engine.get_text_embedding_batch(documents)

    for output in (query_embedding, document_embedding, document_embeddings):
        pprint(output)
        print('=' * 16)


if __name__ == '__main__':
    main()
