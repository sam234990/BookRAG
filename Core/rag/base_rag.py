from abc import ABC, abstractmethod
from Core.provider.llm import LLM
from typing import List, Tuple, Any


class BaseRAG(ABC):
    def __init__(
        self,
        llm: LLM,
        name: str = "Base RAG",
        description: str = "Base Retrieval Augmented Generation",
    ):
        self.llm = llm
        self.name = name
        self.description = description

    @abstractmethod
    def _retrieve(self, query: str, **kwargs):
        pass

    @abstractmethod
    def _create_augmented_prompt(self, query: str) -> str:
        pass

    @abstractmethod
    def generation(self, query: str, query_output_dir: str) -> Tuple[str, List[Any]]:
        """
        Generates an answer for a given query and returns the answer along with the context used.
        Returns:
            Tuple[str, List[Any]]: A tuple contains final answer and the retrieval ids
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        pass
