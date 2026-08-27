# bigram-to-llm

Building a language model from scratch in PyTorch, starting from a bigram and working up to a modern decoder-only transformer.

The `notebooks/` folder is the walkthrough. Each notebook adds one thing to the model before it, and explains the limitation that made the addition necessary.

The `.py` files in the root are the final, runnable pipeline. These will change over time as the model evolves. The notebooks, however, are
frozen in time from when I made them. This means earlier notebooks may 
have issues I resolved later. E.g. in the first 5 notebooks I didn't
move anything onto the device, but subsequent notebooks will.

Currently the models are all trained on Tiny Shakespeare. When I add all
modern additions, I will use a larger data set, a bigger model, and
train for longer to get an LLM. However, I will hold off until I am done
because a training run large enough for an LLM costs time and money.

## Setup

Python 3.13. It'll use CUDA, MPS, or CPU depending on what's available.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running it

Three steps, in order:

```python
from run import preprocess, train, generate

preprocess()   # download the corpus, build the tokenizer, save to artifacts/
train()        # train and save a checkpoint
generate()     # stream 500 characters from the checkpoint
```

Each of the 3 stages is saved onto disk. So once you train you only have to run generate(). Also once you run preprocess(), you don't have to again.

`generate()` picks a random starting character unless you give it a prompt:

```python
from run import generate, GenerateConfig

generate(GenerateConfig(prompt='ROMEO:', tokens_to_gen=1000))
```

Every config is a dataclass, so changing one means passing in a new one. The
other configs work the same way, letting you change the architecture of the
transformer, the training hyperparams, etc.

```python
from run import train
from train import TrainConfig

train(train_config=TrainConfig(total_steps=20_000, lr=3e-4))
```

You can also uncomment what you want at the bottom of [run.py](run.py) and run
`python run.py`.

The defaults are a 2 layer, 8 head model with `d_model=128` and a context size
of 128, trained for 10k steps. They're in [transformer.py](transformer.py) and
[train.py](train.py).

## Files

- [transformer.py](transformer.py). The model: attention, MLP, block, and the
  full transformer.
- [tokenizer.py](tokenizer.py). `CharTokenizer`, plus a protocol so a different
  tokenizer can be dropped in later.
- [get_data.py](get_data.py). Downloading the corpus, splitting it, and sampling
  batches.
- [train.py](train.py). The training loop.
- [generate.py](generate.py). Autoregressive sampling. It yields one token at a
  time so generation can stream.
- [run.py](run.py). Configs and the three entry points above.

Checkpoints save the configs and the tokenizer along with the weights, so
loading one doesn't require knowing how it was trained.

A note on generation: it samples from the softmax instead of taking the argmax.
Greedy decoding on a model this small gets stuck repeating one token forever,
which is a bug I ran into in notebook 01.

## What's next

I will continue to build on this repo. Now that the basic transformer is done, I can start to work on adding modern features.

## License

MIT.
