import json
from collections import Counter


class tokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.bpe_merges = {}

    def train(self, text, vocab_size, special_tokens=()):
        base_tokens = sorted(set(text)) + list(special_tokens)
        self.vocab = dict(enumerate(base_tokens))
        self.inverse_vocab = {token: i for i, token in self.vocab.items()}

        token_ids = [self.inverse_vocab[char] for char in text]

        for new_id in range(len(self.vocab), vocab_size):
            pair = self.most_frequent_pair(token_ids)
            if pair is None:
                break
            token_ids = self.replace_pair(token_ids, pair, new_id)
            self.bpe_merges[pair] = new_id

        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def encode(self, text):
        ids = [self.inverse_vocab[char] for char in text]
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.bpe_merges.get(p, float("inf")))
            if pair not in self.bpe_merges:
                break
            ids = self.replace_pair(ids, pair, self.bpe_merges[pair])
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def save(self, path):
        data = {
            "vocab": [self.vocab[i] for i in range(len(self.vocab))],
            "merges": [[p0, p1, new_id] for (p0, p1), new_id in self.bpe_merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = dict(enumerate(data["vocab"]))
        self.inverse_vocab = {token: i for i, token in self.vocab.items()}
        self.bpe_merges = {(p0, p1): new_id for p0, p1, new_id in data["merges"]}

    @staticmethod
    def most_frequent_pair(token_ids):
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None
        return max(pairs, key=pairs.get)

    @staticmethod
    def replace_pair(token_ids, pair, new_id):
        replaced = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
                replaced.append(new_id)
                i += 2
            else:
                replaced.append(token_ids[i])
                i += 1
        return replaced
