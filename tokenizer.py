import torch
from typing import Protocol

class Tokenizer(Protocol):
    vocabulary: list[str]
    vocab_size: int
    
    def __init__(self, corpus: str) -> None: ...
    def encode(self, text: str) -> torch.Tensor: ...
    def decode(self, tokens) -> str: ...

    def state(self) -> dict: ...

    @classmethod
    def from_state(cls, state: dict) -> 'Tokenizer': ...
    

class CharTokenizer:
    def __init__(self, corpus):
        """
        Args:
            corpus (str): The corpus
        """
        self.vocabulary = sorted(list(set(corpus)))
        self._set_vocabulary(self.vocabulary)
        
    def state(self):
        """Serializes state of tokenizer for later rebuilding

        Returns:
            dict: State of the tokenizer, later passed to from_state
        """
        return {'vocabulary': list(self.vocabulary)}

    @classmethod
    def from_state(cls, state):
        """Rebuilds tokenzier from a previous state

        Args:
            state (dict): The state to rebuild from

        Returns:
            CharTokenizer: The tokenizer instance
        """
        tokenizer = cls.__new__(cls)

        # duplicate list to prevent accidental mutation
        tokenizer._set_vocabulary(list(state['vocabulary']))
        return tokenizer
        
    def _set_vocabulary(self, vocabulary):
        """Helper to set attributes

        Args:
            vocabulary (list[str]): The vocab to create attributes from
        """
        self.vocab_size = len(vocabulary)
        self.vocabulary = vocabulary
        self.stoi = {ch: i for i, ch in enumerate(vocabulary)} 
        self.itos = {i: ch for i, ch in enumerate(vocabulary)}
        
    def encode(self, text):
        """
        Args:
            text (str): The text to encode
        Returns:
            torch.Tensor: 1D tensor of token ids, dtype torch.long
        """
        return torch.tensor(
            [self.stoi[char] for char in text],
            dtype=torch.long
        )

    def decode(self, tokens):
        """
        Args:
            tokens: 1D tensor of token ids OR a python list of token ids
        Returns:
            str: The decoded text
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return ''.join([self.itos[token] for token in tokens])