import torch
import torch.nn as nn

from get_data import load_data, split_data
from tokenizer import CharTokenizer
from train import TrainConfig, train_model
from transformer import TransformerConfig, Transformer

TINY_SHAKESPEARE_FILE_PATH = 'data/tinyshakespeare.txt'
TINY_SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

def get_device():
    """Returns the best available device for training"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    device = get_device()
    print(f'Using device: {device}')

    corpus = load_data(TINY_SHAKESPEARE_FILE_PATH, TINY_SHAKESPEARE_URL)

    tokenizer = CharTokenizer(corpus)
    data = tokenizer.encode(corpus).to(device)

    train_data, val_data, test_data = split_data(data, 0.9, 0.05)

    train_config = TrainConfig()
    transformer_config = TransformerConfig(tokenizer.vocab_size)

    model = Transformer(transformer_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), train_config.lr)
    
    losses, seq_done = train_model(
        model, criterion, optimizer, train_data, train_config
    )
    
if __name__ == '__main__':
    main()