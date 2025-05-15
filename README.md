## Introduction
This library trains _k_-sparse autoencoders (SAEs) and transcoders on the activations of HuggingFace language models, roughly following the recipe detailed in [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093v1) (Gao et al. 2024).

This is a lean, simple library with few configuration options. Unlike most other SAE libraries (e.g. [SAELens](https://github.com/jbloomAus/SAELens)), it does not cache activations on disk, but rather computes them on-the-fly. This allows us to scale to very large models and datasets with zero storage overhead, but has the downside that trying different hyperparameters for the same model and dataset will be slower than if we cached activations (since activations will be re-computed). We may add caching as an option in the future.

Following Gao et al., we use a TopK activation function which directly enforces a desired level of sparsity in the activations. This is in contrast to other libraries which use an L1 penalty in the loss function. We believe TopK is a Pareto improvement over the L1 approach, and hence do not plan on supporting it.

## Loading pretrained SAEs

To load a pretrained SAE from the HuggingFace Hub, you can use the `Sae.load_from_hub` method as follows:

```python
from sparsify import Sae

sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10")
```

This will load the SAE for residual stream layer 10 of Llama 3 8B, which was trained with an expansion factor of 32. You can also load the SAEs for all layers at once using `Sae.load_many`:

```python
saes = Sae.load_many("EleutherAI/sae-llama-3-8b-32x")
saes["layers.10"]
```

The dictionary returned by `load_many` is guaranteed to be [naturally sorted](https://en.wikipedia.org/wiki/Natural_sort_order) by the name of the hook point. For the common case where the hook points are named `embed_tokens`, `layers.0`, ..., `layers.n`, this means that the SAEs will be sorted by layer number. We can then gather the SAE activations for a model forward pass as follows:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
inputs = tokenizer("Hello, world!", return_tensors="pt")

with torch.inference_mode():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    outputs = model(**inputs, output_hidden_states=True)

    latent_acts = []
    for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
        # (N, D) input shape expected
        hidden_state = hidden_state.flatten(0, 1)
        latent_acts.append(sae.encode(hidden_state))

# Do stuff with the latent activations
```

For use cases beyond collecting residual stream SAE activations, we recommend PyTorch hooks ([see examples](https://gist.github.com/luciaquirke/7105708dac0cfc632d68f33c79b59e5c).)

## Training SAEs and transcoders

To train SAEs from the command line, you can use the following command:

```bash
python -m sparsify EleutherAI/pythia-160m <optional dataset>
```
By default, we use the `EleutherAI/SmolLM2-135M-10B` dataset for training, but you can use any dataset from the HuggingFace Hub, or any local dataset in HuggingFace format (the string is passed to `load_dataset` from the `datasets` library).

The CLI supports all of the config options provided by the `TrainConfig` class. You can see them by running `python -m sparsify --help`.

Programmatic usage is simple. Here is an example:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify import SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize

MODEL = "HuggingFaceTB/SmolLM2-135M"
dataset = load_dataset(
    "EleutherAI/SmolLM2-135M-10B", split="train",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer)


gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
)

cfg = TrainConfig(SaeConfig(), batch_size=16)
trainer = Trainer(cfg, tokenized, gpt)

trainer.fit()
```

## Finetuning SAEs

To finetune a pretrained SAE, pass its path to the `finetune` argument.

```bash
python -m sparsify EleutherAI/pythia-160m togethercomputer/RedPajama-Data-1T-Sample --finetune EleutherAI/sae-pythia-160m-32x
```

## Custom hookpoints

By default, the SAEs are trained on the residual stream activations of the model. However, you can also train SAEs on the activations of any other submodule(s) by specifying custom hookpoint patterns. These patterns are like standard PyTorch module names (e.g. `h.0.ln_1`) but also allow [Unix pattern matching syntax](https://docs.python.org/3/library/fnmatch.html), including wildcards and character sets. For example, to train SAEs on the output of every attention module and the inner activations of every MLP in GPT-2, you can use the following code:

```bash
python -m sparsify gpt2 --hookpoints "h.*.attn" "h.*.mlp.act"
```

To restrict to the first three layers:

```bash
python -m sparsify gpt2 --hookpoints "h.[012].attn" "h.[012].mlp.act"
```

We currently don't support fine-grained manual control over the learning rate, number of latents, or other hyperparameters on a hookpoint-by-hookpoint basis. By default, the `expansion_factor` option is used to select the appropriate number of latents for each hookpoint based on the width of that hookpoint's output. The default learning rate for each hookpoint is then set using an inverse square root scaling law based on the number of latents. If you manually set the number of latents or the learning rate, it will be applied to all hookpoints.

## Distributed training

We support distributed training via PyTorch's `torchrun` command. By default we use the Distributed Data Parallel method, which means that the weights of each SAE are replicated on every GPU.

```bash
torchrun --nproc_per_node gpu -m sparsify meta-llama/Meta-Llama-3-8B --batch_size 1 --layers 16 24 --k 192 --grad_acc_steps 8 --ctx_len 2048
```

This is simple, but very memory inefficient. If you want to train SAEs for many layers of a model, we recommend using the `--distribute_modules` flag, which allocates the SAEs for different layers to different GPUs. Currently, we require that the number of GPUs evenly divides the number of layers you're training SAEs for.

```bash
torchrun --nproc_per_node gpu -m sparsify meta-llama/Meta-Llama-3-8B --distribute_modules --batch_size 1 --layer_stride 2 --grad_acc_steps 8 --ctx_len 2048 --k 192 --load_in_8bit --micro_acc_steps 2
```

The above command trains an SAE for every _even_ layer of Llama 3 8B, using all available GPUs. It accumulates gradients over 8 minibatches, and splits each minibatch into 2 microbatches before feeding them into the SAE encoder, thus saving a lot of memory. It also loads the model in 8-bit precision using `bitsandbytes`. This command requires no more than 48GB of memory per GPU on an 8 GPU node.

## TODO

There are several features that we'd like to add in the near future:
- [ ] Support for caching activations
- [ ] Evaluate SAEs with KL divergence when grafted into the model

If you'd like to help out with any of these, please feel free to open a PR! You can collaborate with us in the sparse-autoencoders channel of the EleutherAI Discord.

## Installation

`pip install eai-sparsify`

## Development

Run `pip install -e .[dev]` from the sparsify directory.

We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) for releases.

## Experimental features

Linear k decay schedule:

```bash python -m sparsify gpt2 --hookpoints "h.*.attn" "h.*.mlp.act" --k_decay_steps 10_000```

GroupMax activation function:

```bash python -m sparsify gpt2 --hookpoints "h.*.attn" "h.*.mlp.act" --activation groupmax```

End-to-end training:

```bash python -m sparsify gpt2 --hookpoints "h.*.attn" "h.*.mlp.act" --loss_fn ce```

or

```bash python -m sparsify gpt2 --hookpoints "h.*.attn" "h.*.mlp.act" --loss_fn kl```
