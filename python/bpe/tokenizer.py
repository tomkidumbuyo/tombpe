from pathlib import Path
from collections import Counter
import json
import regex as re 
from collections import defaultdict
from itertools import combinations

class BasicTokenizer:

    PRE_TOKENIZATION_REGEX = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def __init__(self, vocab_size=32768, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.paired_tokens = {}
        self.utf8_vocab = {i: b.decode('utf-8', errors='ignore') for i, b in ((i, bytes([i])) for i in range(256))}

    def _get_pair_frequencies(self, divided_tokens: list[list[int]]) -> dict[tuple, int]:
        result = defaultdict(int)
        for sublist in divided_tokens:
            for i in range(len(sublist) - 1):
                pair = (sublist[i], sublist[i + 1])
                result[pair] += 1
        return {k: v for k, v in result.items() if v >= self.min_frequency}

    def _add_new_paired_tokens(self, pair: str) -> int:
        newPairedTokenId = len(self.utf8_vocab) + len(self.paired_tokens) + 1

        if(pair[0] > len(self.utf8_vocab)): pair = (*self.paired_tokens[pair[0]], pair[1])
        if(pair[len(pair) - 1] > len(self.utf8_vocab)): pair = (*pair[:-1], *self.paired_tokens[pair[len(pair) - 1]])

        key = next((k for k, v in self.paired_tokens.items() if v == pair), None)

        if key: return key
        self.paired_tokens[newPairedTokenId] = pair
        return newPairedTokenId

    def _merge_pairs(self, divided_tokens):
        while True:
            pair_frequencies = self._get_pair_frequencies(divided_tokens)
            if len(pair_frequencies) == 0:
                break
            i = 0
            for pair, count in pair_frequencies.items():
                newId = self._add_new_paired_tokens(pair)
                for i in range(len(divided_tokens)):
                    j = len(divided_tokens[i]) - 2
                    while j >= 0:
                        if divided_tokens[i][j] == pair[0] and divided_tokens[i][j + 1] == pair[1]:
                            divided_tokens[i][j:j+2] = [newId]
                        j -= 1
                        
        all_tokens = [item for sublist in divided_tokens for item in sublist]
        self.paired_tokens = {k: v for k, v in self.paired_tokens.items() if k in all_tokens}            
        return all_tokens

    def _add_unique_characters_to_utf8_vocab(self, text: str):
        invalid_chars = set(text) - {chr(i) for i in range(256)}
        for invalid_char in invalid_chars:
            self.utf8_vocab[len(self.utf8_vocab) + 1] = invalid_char
   
    def train(self, training_string):
        training_string_sections = self.PRE_TOKENIZATION_REGEX.findall(training_string)
        self._get_unique_characters(training_string)
        tokens = list(training_string.encode("utf-8"))
        mergedTokens = self._merge_pairs([list(section.encode("utf-8")) for section in training_string_sections])
        self.paired_tokens = {**self.paired_tokens, **self.paired_tokens}

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
        return ''.join([self.vocab[token] for token in tokens])

    def _get_vocab_from_paired_tokens(self):
        self.vocab = self.utf8_vocab
        for token_id, pair in self.paired_tokens.items():
            self.vocab[token_id] = ''.join([self.vocab[p] for p in pair])
        self.vocab = dict(sorted(((k, v) for k, v in self.vocab.items() if v != ""), key=lambda x: len(x[1]), reverse=True))
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