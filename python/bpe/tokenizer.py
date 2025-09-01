from pathlib import Path
from collections import Counter

class BasicTokenizer:
    
    def __init__(self, vocab_size=32768, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.paired_tokens = {}

    def _get_pair_frequencies(self, tokens, min_frequency=1):
        pair_frequencies = dict(Counter(zip(tokens, tokens[1:])))
        return {pair: count for pair, count in pair_frequencies.items() if count >= min_frequency}

    def _add_new_paired_tokens(self, pair: str):
        newPairedTokenId = len(self.paired_tokens) + 256
        if(pair[0] >= 256): pair = (*self.paired_tokens[pair[0]], pair[1])
        if(pair[len(pair) - 1] >= 256): pair = (*pair[:-1], *self.paired_tokens[pair[len(pair) - 1]])
        key = next((k for k, v in self.paired_tokens.items() if v == pair), None)
        if key: return key
        self.paired_tokens[newPairedTokenId] = pair
        return newPairedTokenId


    def _merge_pairs(self, tokens):
        while True:
            pair_frequencies = self._get_pair_frequencies(tokens, 2)
            if len(pair_frequencies) == 0:
                break
            i = 0
            for pair, count in pair_frequencies.items():
                newId = self._add_new_paired_tokens(pair)
                while i < len(tokens) - 1:
                    if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                        tokens[i:i+2] = [newId]
                    else:
                        i += 1 
        return tokens


    def train(self, training_string):
        tokens = list(training_string.encode("utf-8"))
        mergedTokens = self._merge_pairs(tokens)
        


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