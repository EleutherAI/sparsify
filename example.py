import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify import SaeConfig, TrainConfig, Trainer
from sparsify.data import chunk_and_tokenize

MODEL = "EleutherAI/pythia-70m"
train_split = "train[:2%]"
eval_split = "train[:1%]"

print("Loading raw datasets...")
train_raw = load_dataset("NeelNanda/pile-10k", split=train_split)
eval_raw = load_dataset("NeelNanda/pile-10k", split=eval_split)
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
tokenizer.model_max_length = 64
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizing datasets...")
train_ds = chunk_and_tokenize(train_raw, tokenizer, max_seq_len=64)
eval_ds = chunk_and_tokenize(eval_raw, tokenizer, max_seq_len=64)

print(f"Train dataset size: {len(train_ds)}")
print(f"Eval dataset size: {len(eval_ds)}")

# ---------------------------
# Load tiny GPT (Pythia-70M)
# ---------------------------
gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.float32,
)

# ---------------------------
# Training config
# ---------------------------
cfg = TrainConfig(
    sae=SaeConfig(),
    batch_size=1,
    grad_acc_steps=1,
    micro_acc_steps=1,
    loss_fn="kl",
    wandb_log_frequency=50,
    save_every=100,
    run_name="tiny-gpt2-kl",
    log_to_wandb=False,
)

# ---------------------------
# Trainer
# ---------------------------
trainer = Trainer(cfg, train_ds, gpt, eval_dataset=eval_ds)
trainer.fit()
