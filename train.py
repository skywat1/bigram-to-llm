import tqdm
from dataclasses import dataclass

from get_data import get_batch

@dataclass
class TrainConfig:
    batch_size: int = 64
    seq_len: int = 128
    total_steps: int = 10_000
    lr: float = 1.e-3

def train_model(model, criterion, optimizer, data, config: TrainConfig):
    """Trains a model for config.total_steps on randomly sampled batches

    Args:
        model: Model maps token ids of shape (batch_size, seq_len) to 
            logits of shape (batch_size, seq_len, vocab_size)
        criterion: The loss function
        optimizer: The optimizer holding the model's parameters
        data: 1D tensor of torch.long containing the corpus tokens
        config (TrainConfig): Config for training

    Returns:
        losses: A list of losses at each step
        seq_done: Total number of sequences trained on
    """
    model.train()
    losses = []
    seq_done = 0

    pbar = tqdm.tqdm(range(config.total_steps))
    for _ in pbar:
        x, y = get_batch(data, config.batch_size, config.seq_len)
        optimizer.zero_grad()
        
        logits = model(x)
        logits = logits.reshape(-1, logits.size(-1))
        y = y.reshape(-1)
        
        loss = criterion(logits, y)
        loss.backward()
        
        losses.append(loss.item())
        seq_done += config.batch_size
        
        pbar.set_description(f'Loss: {loss.item():.4f}, Sequences Done: {seq_done}')
        
        optimizer.step()
        
    return losses, seq_done
    