from pathlib import Path
from collections import Counter
import json

class BasicTokenizer:
    
    def __init__(self, vocab_size=32768, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.paired_tokens = {}

    def _get_pair_frequencies(self, tokens, min_frequency=1) -> dict[tuple, int]:
        pair_frequencies = dict(Counter(zip(tokens, tokens[1:])))
        return {pair: count for pair, count in pair_frequencies.items() if count >= min_frequency}

    def _add_new_paired_tokens(self, pair: str) -> int:
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
        self._get_vocab_from_paired_tokens()

    def encode(self, text: str) -> list[int]:
        result = []
        i = 0
        while i < len(text):
            match = None
            for key, sub in self.vocab.items():
                if text.startswith(sub, i):
                    match = [key, sub]
                    break
            if match:
                result.append(match[0])
                i += len(match[1])
            else:
                result.append(text[i])
                i += 1
        return result

    def decode(self, tokens: list[int]) -> str:
        print(tokens)
        return ''.join([self.vocab[token] for token in tokens])

    def _get_vocab_from_paired_tokens(self):
        self.vocab = {i: b.decode('utf-8', errors='ignore') 
              for i, b in ((i, bytes([i])) for i in range(256)) 
              if b.decode('utf-8', errors='ignore') != ''}
        for token_id, pair in self.paired_tokens.items():
            self.vocab[token_id] = ''.join([self.vocab[p] for p in pair])
        self.vocab = dict(sorted(self.vocab.items(), key=lambda x: len(x[1]), reverse=True))
        return self.vocab

    def save(self, output_path: str):
        empty_bpe_model_json_path = Path(__file__).parent.parent / "assets" / "empty_bpe_model.json"

        with empty_bpe_model_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        data["model"]["vocab"] =  self._get_vocab_from_paired_tokens()

        with Path(output_path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self, path: str):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = {int(k): v for k, v in data["model"]["vocab"].items()}