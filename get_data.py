import torch
import requests
import os

def load_data(file_path, data_url=None):
    """Fetches the data file and saves it as a string in a text file

    Args:
        file_path (str): The file path to save the data to, or read the data
            from if it already exists
        data_url (str, optional): The url to get data from. Defaults to None.

    Returns:
        str: The corpus as a string
    """
    exists = os.path.exists(file_path)
    
    if not exists and data_url is None:
        raise ValueError('data_url must be provided for file_path that does not exist')
    
    if not exists:
        with open(file_path, 'w') as f:
            f.write(requests.get(data_url).text) # type: ignore
        
    with open(file_path, 'r') as f:
        corpus = f.read()
    return corpus
    

def split_data(data, train_frac, val_frac):
    """Splits data into train, cv, and test sets

    Args:
        data: A 1D tensor of the data 
        train_frac (float): The fraction of data to give to test
        val_frac (float): The fraction of data to give to validatioin
        
    Returns:
        A tuple of train, val, and test data
    """
    if train_frac < 0 or val_frac < 0 or train_frac + val_frac > 1:
        raise ValueError('train_frac and val_frac cannot sum to > 1 and must both be positive')

    n = len(data)
    i = int(train_frac * n)
    j = i + int(val_frac * n)
    
    train_data = data[:i]
    val_data = data[i:j]
    test_data = data[j:]
    
    return train_data, val_data, test_data


def get_batch(data, batch_size: int, seq_len: int):
    """Get a batch of (x, y) data from the corpus

    Args:
        data: 1D tensor of torch.long containing the corpus tokens
        batch_size (int): The number of samples to get
        seq_len (int): The length of each sequence to fetch
        
    Returns:
        X: a 2D tensor of shape (batch_size, seq_len) containing token IDs
        Y: a 2D tensor of shape (batch_size, seq_len) containing token IDs
    """
    n = len(data)
    if seq_len > n:
        raise ValueError('seq_len is larger than size of corpus')
    
    start_idxs = torch.randint(n - seq_len, (batch_size,), device=data.device)
    start_idxs = start_idxs[:, None]
    range_idxs = torch.arange(seq_len + 1, device=data.device)
    idxs = start_idxs + range_idxs
    
    values = data[idxs]
    X, Y = values[:, :-1], values[:, 1:]
    return X, Y