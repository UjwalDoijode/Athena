import re
from typing import List
from .base import BaseChunker, Chunk

class SentenceChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 512, overlap_sentences: int = 1):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    def chunk(self, text: str) -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks, current, current_size, index, char_pos = [], [], 0, 0, 0

        for sentence in sentences:
            slen = len(sentence)
            if current_size + slen > self.max_chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(self._make_chunk(chunk_text, index,
                    char_pos - current_size, char_pos, chunker="sentence"))
                index += 1
                current = current[-self.overlap_sentences:] if self.overlap_sentences else []
                current_size = sum(len(s) for s in current)
            current.append(sentence)
            current_size += slen
            char_pos += slen + 1

        if current:
            chunk_text = " ".join(current)
            chunks.append(self._make_chunk(chunk_text, index,
                char_pos - current_size, char_pos, chunker="sentence"))
        return chunks