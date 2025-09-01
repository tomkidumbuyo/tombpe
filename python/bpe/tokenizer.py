from pathlib import Path
from collections import Counter

class BasicTokenizer:
    
    def __init__(self, vocab_size=32768, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    def _get_pair_frequencies(self, tokens, min_frequency=1):
        pair_frequencies = dict(Counter(zip(tokens, tokens[1:])))
        return {pair: count for pair, count in pair_frequencies.items() if count >= min_frequency}

    def _merge_pairs(self, tokens):
        newIds = []
        while True:
            pair_frequencies = self._get_pair_frequencies(tokens, 2)
            if len(pair_frequencies) == 0:
                break
            i = 0
            for pair, count in pair_frequencies.items():
                newId = len(newIds) + 256
                newIds.append({newId: pair})
                while i < len(tokens) - 1:
                    if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                        tokens[i:i+2] = [newId]
                    else:
                        i += 1 
        return newIds, tokens

    def train(self, training_string):
        tokens = list(training_string.encode("utf-8"))
        newIds, mergedTokens = self._merge_pairs(tokens)
        self.tokens = newIds
 
        


    def encode(self, text: str) -> list[int]:
        pass

    def decode(self, tokens: list[int]) -> str:
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")