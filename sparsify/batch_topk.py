from sparsify import SparseCoder
from sparsify.data import chunk_and_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch


from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-8M")

dataset = load_dataset("roneneldan/TinyStories")

tokens = chunk_and_tokenize(dataset["train"], tokenizer, max_seq_len=256)

sae = SparseCoder.load_from_disk("tinystories-8m-batch-test-2/h.4.mlp", device="cuda")

from sparsify.collect_activations import collect_activations
with torch.inference_mode():
    for i in range(10):
        data = tokens[i:i+1]

        with collect_activations(model, ["transformer.h.4.mlp"]) as activations:
            model(**data)

        out = sae.forward(activations["transformer.h.4.mlp"].cuda().flatten(0, 1))
        print("fvu", out.fvu)

    # latent_acts = []

    # for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
    #     latent_acts.append(sae.encode(hidden_state))

# Do stuff with the latent activations