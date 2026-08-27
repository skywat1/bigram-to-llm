import torch
import torch.nn as nn
import os
from dataclasses import dataclass, asdict

from get_data import load_data, split_data
from tokenizer import Tokenizer, CharTokenizer
from train import TrainConfig, train_model
from transformer import TransformerConfig, Transformer
from generate import gen_tokens

TINY_SHAKESPEARE_FILE_PATH = 'data/tinyshakespeare.txt'
TINY_SHAKESPEARE_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
SAVED_DATASET_PATH = 'artifacts/dataset.pt'
CHECKPOINT_PATH = 'artifacts/model.pt'

def get_device():
    """Returns the best available device for training"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

@dataclass
class PreprocessConfig:
    tokenizer_cls: type[Tokenizer] = CharTokenizer
    corpus_file_path: str = TINY_SHAKESPEARE_FILE_PATH
    data_url: str = TINY_SHAKESPEARE_URL
    saved_dataset_path: str = SAVED_DATASET_PATH

def preprocess(config=None):
    """Preprocesses the data and saves it to disk

    Args:
        config (PreprocessConfig, optional): The config settings. 
            Defaults to None
    Returns:
        tokenizer: The tokenizer that was created
        data: A 1d tensor of token IDs from the corpus
    """
    config = config or PreprocessConfig()
    corpus = load_data(config.corpus_file_path, config.data_url)
    tokenizer = config.tokenizer_cls(corpus)
    data = tokenizer.encode(corpus)
    os.makedirs(os.path.dirname(config.saved_dataset_path), exist_ok=True)
    torch.save(
        {'tokenizer': tokenizer.state(), 'data': data},
        config.saved_dataset_path
    )
    
    return tokenizer, data

def load_dataset(config=None):
    """Loads saved data from preprocess step. Preprocess MUST have ran first

    Args:
        config (PreprocessConfig, optional): The config settings. 
            Defaults to None
    Returns:
        tokenizer: The saved tokenizer
        data: Saved 1d tensor of token IDs
    """
    config = config or PreprocessConfig()
    if not os.path.exists(config.saved_dataset_path):
        raise FileNotFoundError('Dataset path does not exist')
    
    saved = torch.load(config.saved_dataset_path, weights_only=True)
    tokenizer = config.tokenizer_cls.from_state(saved['tokenizer'])
    return tokenizer, saved['data']


@dataclass
class TrainRunConfig:
    train_frac: float = 0.9
    val_frac: float = 0.05
    checkpoint_path: str = CHECKPOINT_PATH

def train(preprocess_config=None, run_config=None, train_config=None):
    """Trains a transformer on preprocessed corpus and saves checkpoint

    Args:
        preprocess_config (PreprocessConfig, optional): MUST match the one used
            in preprocess
        run_config (TrainRunConfig, optional): data split fractions and save 
            path
        train_config (TrainConfig): Training hyperparameters

    Returns:
        model: The trained model
        losses: A list of losses at each step
    """
    preprocess_config = preprocess_config or PreprocessConfig()
    run_config = run_config or TrainRunConfig()
    train_config = train_config or TrainConfig()

    device = get_device()
    print(f'Using device: {device}')

    tokenizer, data = load_dataset(preprocess_config)
    data = data.to(device)
    train_data, val_data, test_data = split_data(
        data, run_config.train_frac, run_config.val_frac
    )

    transformer_config = TransformerConfig(tokenizer.vocab_size)
    model = Transformer(transformer_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), train_config.lr)

    losses, seq_done = train_model(
        model, criterion, optimizer, train_data, train_config
    )

    os.makedirs(os.path.dirname(run_config.checkpoint_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'transformer_config': asdict(transformer_config),
        'train_config': asdict(train_config),
        'run_config': asdict(run_config),
        'tokenizer': tokenizer.state(),
        'losses': losses,
        'seq_done': seq_done,
    }, run_config.checkpoint_path)

    return model, losses


@dataclass
class GenerateConfig:
    tokenizer_cls: type[Tokenizer] = CharTokenizer
    checkpoint_path: str = CHECKPOINT_PATH
    tokens_to_gen: int = 500
    prompt: str | None = None
    
def load_checkpoint(config=None, device=None):
    """Rebuilds the tokenizer and model saved by train

    Args:
        config (GenerateConfig, optional): Where the checkpoint lives
        device (torch.device, optional): Device to put the model on

    Returns:
        model: The trained model, ready to generate with
        tokenizer: The tokenizer the model was trained with
    """
    config = config or GenerateConfig()
    device = device or get_device()

    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError('Checkpoint path does not exist')

    saved = torch.load(
        config.checkpoint_path, map_location=device, weights_only=True
    )

    tokenizer = config.tokenizer_cls.from_state(saved['tokenizer'])

    transformer_config = TransformerConfig(**saved['transformer_config'])
    model = Transformer(transformer_config).to(device)
    model.load_state_dict(saved['model_state_dict'])

    return model, tokenizer

def generate(config=None):
    """Streams generated text from a trained checkpoint

    Args:
        config (GenerateConfig, optional): Checkpoint, length and prompt

    Returns:
        str: The generated text
    """
    config = config or GenerateConfig()

    model, tokenizer = load_checkpoint(config)

    pieces = []
    for piece in gen_tokens(
        model, tokenizer, config.tokens_to_gen, config.prompt
    ):
        print(piece, end='', flush=True)
        pieces.append(piece)
    print()

    return ''.join(pieces)

if __name__ == '__main__':
    # preprocess()
    # train()
    generate()