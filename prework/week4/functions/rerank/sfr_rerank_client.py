import json
from typing import List, Union, Optional
from typing import Sequence

from pydantic import BaseModel

import requests

# Define the request and response models for the reranker client
# Approximates Cohere schema: https://github.com/cohere-ai/cohere-python/blob/6fdfe5ca3c776bd628eba34d6c8db51dfd6fe73b/src/cohere/base_client.py#L1126

class RerankRequestDocumentsItemText(BaseModel):
    text: str

RerankRequestDocumentsItem = Union[str, RerankRequestDocumentsItemText]

class RerankResponseResultsItemDocument(BaseModel):
    text: str

class RerankResponseResultsItem(BaseModel):
    document: RerankResponseResultsItemDocument
    index: int # Original index of the document
    relevance_score: float # Relevance score of the document

class RerankResponse(BaseModel):
    id: Optional[str]
    results: List[RerankResponseResultsItem]


class SFRRerankClient:
    """Client for internally hosted reranking model."""

    def rerank(
        self,
        query: str,
        documents: Sequence[RerankRequestDocumentsItem],
        top_n: int = 3,
        model: str = "mistral-instruct"
    ) -> RerankResponse:
        """
        Rerank documents based on relevance to the query

        Args:
            query (str): Query to rerank documents by
            documents (List[RerankRequestDocumentsItem]): List of documents to rerank. Each document must be either (1) a string,
                or (2) an object with a `text` attribute
            top_n (int): Number of documents to return

        Returns:
            List of reranked documents and associated data (RerankResponse):
        """

        if model != "mistral-instruct":
            raise ValueError(f"Model {model} not supported")

        texts = [doc if isinstance(doc, str) else doc.text for doc in documents]

        args = {
            'query': query,
            'passages': texts,
            'model_name': model
        }
        response = requests.post(
            url='http://reranker.salesforceresearch.ai/rerank',
            json=args
        )
        response_data = json.loads(response.content)
        results = response_data['reranked_passages'][:top_n]
        return RerankResponse(
            id=None,
            results=[
                RerankResponseResultsItem(
                    document=RerankResponseResultsItemDocument(
                        text=result['passage']
                    ),
                    index=result['index'],
                    relevance_score=result['score']
                ) for result in results
            ]
        )

if __name__ == "__main__":
    import pprint
    client = SFRRerankClient()

    # Example query and passages
    query = "What is the capital of the United States?"
    documents = [
        "Carson City is the capital city of the American state of Nevada. At the  2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
        "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. ",
        "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
        "North Dakota is a state in the United States. 672,591 people lived in North Dakota in the year 2010. The capital and seat of government is Bismarck."
    ]

    results = client.rerank(query=query, documents=documents, top_n=3)
    print(results)
    pprint.pprint(results.model_dump())