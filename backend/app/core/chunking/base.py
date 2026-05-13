from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

@dataclass
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

class BaseChunker(ABC):

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Chunk the input text and return a list of Chunk objects."""
        pass

    def _make_chunk(self, text, index, start, end, **meta) -> Chunk:
        return Chunk(
            text=text.strip(), index=index, start_char=start, end_char=end, metadata=meta
        )