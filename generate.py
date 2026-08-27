import torch
import torch.nn.functional as F
import random

@torch.no_grad
def gen_tokens(model, tokenizer, tokens_to_gen, prompt=None):
    model.eval()
    device = next(model.parameters()).device
    
    if prompt is None:
        prompt = tokenizer.vocabulary[
            random.randint(0, tokenizer.vocab_size-1)
        ]
        tokens_to_gen -= 1
        
    prompt_tokens = tokenizer.encode(prompt).to(device)
        
    # Create buffer for the tokens
    n = len(prompt_tokens)
    tokens = torch.empty(n + tokens_to_gen, dtype=torch.long, device=device)
    tokens[:n] = prompt_tokens
    
    for cur_idx in range(n, n + tokens_to_gen):
        context = tokens[max(0, cur_idx-model.context_size) : cur_idx]
        logits = model(context.unsqueeze(0))
        logits = logits[0, -1, :] # (vocab,)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1) # Sampling
        tokens[cur_idx] = token.item()
        
        yield tokenizer.decode(tokens[cur_idx : cur_idx+1])