import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass 
class TransformerConfig:
    vocab_size: int # Supplied from the tokenizer
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 2
    context_size: int = 128
    
    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError('num_heads must divide d_model')


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model (int): Length of the hidden vector after embedding
            num_heads (int): The number of attention heads
                precondition: must perfectly divide d_model
        """
        super().__init__()
        
        # Check preconditions
        if d_model % num_heads != 0:
            raise ValueError('num_heads must divide d_model')
        
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)   

    def forward(self, hidden):
        """
        Args:
            hidden: Input tensor of shape (batch_size, seq_len, d_model) that
            contains the hidden states for each token
            
        Returns:
            A tensor of shape (batch_size, seq_len, d_model) of hidden states
        """
        batch_size, seq_len, _ = hidden.shape
        
        # Construct q, k, v matrices
        q = self.w_q(hidden) # (batch_size, seq_len, d_model)
        k = self.w_k(hidden)
        v = self.w_v(hidden)
        
        # Reshape to (batch_size, seq_len, num_heads, head_size)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_size)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_size)
        
        # Put leading dims at the front 
        # (batch_size, num_heads, seq_len, head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Similarity matrix (batch_size, num_heads, seq_len, seq_len)
        sim = q @ k.transpose(-2, -1)
        sim = self.head_size ** -0.5 * sim
        
        # Apply causal mask
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=hidden.device)
        ) # Lower triangular
        sim = sim.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        sim = F.softmax(sim, dim=-1)
        
        # Get output 
        out = sim @ v # (batch_size, num_heads, seq_len, head_size)
        
        # Reshape
        out = out.transpose(1, 2) # (batch_size, seq_len, num_heads, head_size)
        out = out.reshape(batch_size, seq_len, -1)
        
        # Mix heads
        out = self.w_o(out) # (batch_size, seq_len, d_model)
        return out
    

class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        """
        Args: 
            d_model (int): Length of the hidden vector after embedding
            hidden_dim (int): The width of the hidden layer
                defaults to 4 * d_model
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = 4 * d_model
        
        self.fc = nn.Linear(d_model, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) that
            contains the hidden states for each token
            
        Returns:
            A tensor of shape (batch_size, seq_len, d_model) of hidden states
        """
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Args: 
            d_model (int): Length of the hidden vector after embedding
            num_heads (int): The number of attention heads
                precondition: must perfectly divide d_model
        """
        super().__init__()
        
        self.multihead_attention = MultiheadAttention(d_model, num_heads)
        self.mlp = MLP(d_model)
        
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) that
            contains the hidden states for each token
            
        Returns:
            A tensor of shape (batch_size, seq_len, d_model) of hidden states
        """
        x = x + self.multihead_attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Args: 
            config (TransformerConfig): The model hyperparameters, holding
                vocab_size, d_model, num_heads, num_layers and context_size
        """
        super().__init__()
        
        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional Encoder
        self.context_size = config.context_size
        self.pos_encoder = nn.Embedding(config.context_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.blocks.append(
                TransformerBlock(config.d_model, config.num_heads)
            )
            
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Unembedding
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len) that contains
            the token ids for each token
            
        Returns:
            A tensor of shape (batch_size, seq_len, vocab_size) of logits
        """
        # Check seq_len size
        if x.shape[1] > self.context_size:
            raise ValueError('The sequence length is larger than the max context size')
        
        # Get shape
        batch_size, seq_len = x.shape
        
        # Embedding
        x_embeded = self.embed(x) # (batch_size, seq_len, d_model)
        
        # Positional Encoding
        positions = torch.arange(seq_len, device=x.device)
        positions_encoded = self.pos_encoder(positions)
        
        # Combine
        hidden = x_embeded + positions_encoded # (batch_size, seq_len, d_model)
        
        # Transformer
        for block in self.blocks:
            hidden = block(hidden)
            
        # Final layer norm
        hidden = self.ln_f(hidden)
        
        # Unembedding
        return self.unembed(hidden)