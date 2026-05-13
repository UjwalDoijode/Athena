from typing import List
from .base import BaseChunker, Chunk

class FixedChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        chunks, start, index = [], 0, 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(self._make_chunk(chunk_text, index, start, end, chunker="fixed", chunk_size=self.chunk_size, overlap=self.overlap ))
                index += 1
            start += self.chunk_size - self.overlap
        return chunks