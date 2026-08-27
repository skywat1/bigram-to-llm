import torch

class CharTokenizer:
    def __init__(self, corpus):
        """
        Args:
            corpus (str): The corpus
        """
        unique_chars = sorted(list(set(corpus)))
        
        self.vocab_size = len(unique_chars)
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)} 
        self.itos = {i: ch for i, ch in enumerate(unique_chars)}
        
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