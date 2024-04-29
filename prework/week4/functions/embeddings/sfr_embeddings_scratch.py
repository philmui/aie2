from dataclasses import dataclass
import requests
import json


@dataclass
class SFREmbeddings:
    address: str = 'https://rag.salesforceresearch.ai'
    model_id: str = 'sfr_embedding_mistral'

    def get_query_embedding(self, query: str):
        content = {
            'text': query,
            'model_id': self.model_id,
            'is_query': 'True',
        }
        response = requests.post(
            url=f'{self.address}/embedding',
            json=content,
        )
        output = self.parse_response(response)
        return output

    def get_text_embedding(self, document: str):
        content = {
            'text': document,
            'model_id': self.model_id,
            'is_query': 'False',
        }
        response = requests.post(
            url=f'{self.address}/embedding',
            json=content,
        )
        output = self.parse_response(response)
        return output

    def retrieve(self, query: str, topk: int = 2):
        content = {
            'text': query,
            'model_id': self.model_id,
            'topk': str(topk),
        }
        response = requests.post(
            url=f'{self.address}/xgen_banking_retrieve',
            json=content,
        )
        output = self.parse_response(response)
        return output

    @staticmethod
    def parse_response(response):
        return json.loads(response.content)
