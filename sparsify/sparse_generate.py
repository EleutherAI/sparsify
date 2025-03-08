import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sparsify import SparseCoder
from sparsify.edit_sparse import edit
from datasets import load_dataset

# Make deterministic
torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")

model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-8M")
model.generation_config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

hookpoints = [
    # "transformer.h.0.mlp",
    # "transformer.h.1.mlp",
    # "transformer.h.2.mlp",
    # "transformer.h.3.mlp",
    # "transformer.h.4.mlp",
    "transformer.h.5.mlp",
    # "transformer.h.6.mlp",
    # "transformer.h.7.mlp",
]

sparse_models = {}
for hookpoint in hookpoints:
    disk_hookpoint = hookpoint.replace("transformer.", "")
    sparse_models[hookpoint] = SparseCoder.load_from_disk(
        f"tinystories-8m-sml-batch/{disk_hookpoint}"
    )

dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

# Tokenize the first 3 samples
prefixes = tokenizer(dataset.shuffle(seed=42)["text"][-8192:], return_tensors="pt", padding=True)['input_ids'][:, :15]

for temperature in [0.1]: #  0.7
    print(f"\n\033[1m=== With sparsification (temperature {temperature}) ===\033[0m")
    with torch.inference_mode():
        with edit(
            model,
            hookpoints=hookpoints,
            sparse_models=sparse_models,
        ):
            output = model.generate(
                prefixes, max_new_tokens=20, temperature=temperature, do_sample=True
            )
            for i in range(20):
                print(tokenizer.decode(output[i].tolist()))

    # print(f"\n\033[1m=== Without sparsification (temperature {temperature}) ===\033[0m")
    # with torch.inference_mode():
    #     output = model.generate(
    #         **inputs, max_new_tokens=300, temperature=temperature, do_sample=True
    #     )
    #     print(tokenizer.decode(output[0].tolist()))
