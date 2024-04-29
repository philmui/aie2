import os
from typing import Any, List, Optional

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks import CBEventType, EventPayload
from llama_index.legacy.postprocessor.types import BaseNodePostprocessor
from llama_index.legacy.schema import NodeWithScore, QueryBundle

from functions.rerank.sfr_rerank_client import SFRRerankClient

class SFRRerank(BaseNodePostprocessor):
    model: str = Field(description="Model name.")
    top_n: int = Field(description="Top N nodes to return.")

    _client: SFRRerankClient = SFRRerankClient()

    def __init__(
        self,
        top_n: int = 10,
        model: str = "mistral-instruct",
    ):
        super().__init__(top_n=top_n, model=model)
        if model != "mistral-instruct":
            raise ValueError("Model not supported.")

    @classmethod
    def class_name(cls) -> str:
        return "SFRRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
    
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            texts = [node.node.get_content() for node in nodes]

            results = self._client.rerank(
                model=self.model,
                top_n=self.top_n,
                query=query_bundle.query_str,
                documents=texts,
            ).results

            new_nodes = []
            for result in results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result.index].node, score=result.relevance_score
                )
                new_nodes.append(new_node_with_score)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes

if __name__ == "__main__":
    rerank = SFRRerank(top_n=2)

