#%%
%env CUDA_VISIBLE_DEVICES=7
device = "cuda:0"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# model_name = "HuggingFaceTB/SmolLM2-135M"
model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device
)
tokenizer.pad_token = tokenizer.eos_token
#%%
from datasets import load_dataset
from functools import partial
from collections import OrderedDict, defaultdict
from tqdm.auto import tqdm
from more_itertools import chunked
from itertools import islice
import numpy as np
import torch
class ActivationLoader(object):
    def __init__(self, model, tokenizer, bs=16, msl=128):
        self.model, self.tokenizer = model, tokenizer
        self.ds = load_dataset("HuggingFaceFW/fineweb", split="train", name="sample-10BT")
        self.bs = bs
        self.msl = msl
    
    @torch.inference_mode()
    def __call__(self, take_n, skip_n=0,
                 hooks=None):
        for m in model.modules():
            m._forward_hooks = OrderedDict()
        cache = defaultdict(list)
        if hooks is not None:
            main_module = getattr(self.model, "gpt_neox", getattr(self.model, "model", None))
            for layer_idx, module_name in hooks:
                layer = main_module.layers[layer_idx]
                module = (
                    layer.mlp if module_name == "mlp" else 
                    layer if module_name == "res" else
                    1/0
                )
                module.register_forward_hook(
                    lambda m, i, o:
                        cache[(layer_idx, module_name)]
                        .append((
                            (i if not isinstance(i, tuple) else i[0]),
                            (o if not isinstance(o, tuple) else o[0])
                        )))
        for batch_idx, batch in enumerate(chunked(islice(self.ds, skip_n, skip_n+take_n), self.bs)):
            batch = self.tokenizer.batch_encode_plus(
                [x["text"] for x in batch],
                max_length=self.msl, padding=True, truncation=True,
                return_tensors="pt"
            )
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].bool().cuda()
            self.model(input_ids, output_hidden_states=False, return_dict=False)
            yield {
                k: tuple(
                    t
                    [..., 1:, :].contiguous()
                    .reshape(-1, t.shape[-1])
                    [attention_mask[..., 1:].ravel()]
                    for t in v[0]
                ) for k, v in cache.items()}
            cache.clear()
        for m in model.modules():
            m._forward_hooks = OrderedDict()
        return cache

def strip_high_cossim(x, threshold=0.8, remove_pct = 0.5):
    x = x - x.mean(0)
    x = x / x.norm(dim=-1, keepdim=True)
    has_duplicate = ((x @ x.T - 2 * torch.eye(x.shape[0], device=x.device, dtype=x.dtype)) > threshold).any(dim=-1)
    rand_mask = (torch.rand_like(has_duplicate, dtype=torch.float16) < remove_pct)
    return ~(has_duplicate & rand_mask)
# %%
%load_ext autoreload
%autoreload 2
from sparsify.itda import ITDAConfig, ITDA
import torch
import wandb
torch.set_grad_enabled(False)
mlp_in_out_cache = []
layer = 9
hook = "mlp"
bs = 16
msl = 128
dtype = torch.float32
loader = ActivationLoader(model, tokenizer, bs=bs, msl=msl)
d_model = model.config.hidden_size
add_error = False
subtract_mean = True
skip_connection = False
preprocessing_batches = 64
prune_every = 100
prune_every = 0
# prune_thresh = 0.7
transcode = False
itda_config = ITDAConfig(
    d_model=d_model,
    target_l0=32,
    loss_threshold=0.2,
    add_error=add_error,
    subtract_mean=subtract_mean,
    skip_connection=skip_connection,
    preprocessing_steps=preprocessing_batches,
    error_k=8,
)
itda = ITDA(itda_config, dtype=dtype, device=device)
run_name = "pythia" if "pythia" in model_name else "smollm"
run_name += f"-l{layer}_{hook}"
if transcode:
    run_name += "-transcoder"
if add_error:
    run_name += "-error"
if subtract_mean:
    run_name += "-mean"
if skip_connection:
    run_name += "-skip"
if prune_every > 1:
    run_name += f"-prune{prune_every}"
run_name += f"-k{itda_config.target_l0}"
run = wandb.init(project="itda", entity="eleutherai", name=run_name)
run.config.update(itda_config)
losses = []
dictionary_sizes = []
take_n = 20_000
lim_dictionary_size = 50_000
try:
    for batch_idx, batch in (
        bar := tqdm(enumerate(
            loader(take_n=take_n, hooks=[(layer, hook)])),
                    total=take_n//bs)):
        x, y = batch[(layer, hook)]
        x, y = x.view(-1, d_model), y.view(-1, d_model)
        out = itda.step(x if transcode else y, y)
        if out is None:
            continue
        loss = out.losses.mean().item()
        bar.set_postfix(
            loss=loss,
            dictionary_size=itda.dictionary_size
        )
        losses.append(loss)
        dictionary_sizes.append(itda.dictionary_size)
        wandb.log({
            "loss": loss,
            "dictionary_size": itda.dictionary_size
        }, step=batch_idx)
        if itda.dictionary_size > lim_dictionary_size:
            break
        
        if prune_every and batch_idx % prune_every == prune_every - 1:
            xs = itda.xs[:itda.dictionary_size]
            mask = strip_high_cossim(xs, threshold=prune_thresh)
            itda.xs.data = xs[mask]
            itda.ys.data = itda.ys[:itda.dictionary_size][mask]
            itda.dictionary_size = itda.xs.shape[0]
            wandb.log({
                "pruned_pct": 1 - mask.float().mean().item()
            }, step=batch_idx)
except KeyboardInterrupt:
    pass
wandb.finish()
#%%
from matplotlib import pyplot as plt
skip = 10
plt.xlabel("Step")
plt.ylabel("FVU")
plt.loglog(np.arange(skip, len(losses)), losses[skip:])
plt.xlim(skip, len(losses))
# plt.ylim(0.1, 1.0)
plt.show()
plt.xlabel("Step")
plt.ylabel("Dictionary size")
plt.loglog(dictionary_sizes)
plt.show()
# %%
inputs, outputs = [], []
for batch_idx, batch in enumerate(loader(take_n=100, skip_n=take_n, hooks=[(layer, hook)])):
    x, y = (x for x in batch[(layer, hook)])
    x, y = x.view(-1, d_model), y.view(-1, d_model)
    inputs.append(x)
    outputs.append(y)
inputs, outputs = torch.cat(inputs), torch.cat(outputs)
recon = itda(inputs if transcode else outputs, outputs).y_reconstructed
l2_loss = (recon - outputs).pow(2).sum(-1).mean()
total_variance = (outputs - outputs.mean(0)).pow(2).sum(-1).mean()
fvu = l2_loss / total_variance
fvu
# %%
itda.save_to_disk(f"checkpoints/itda/{run_name}")
#%%
itda = ITDA.load_from_disk(f"checkpoints/itda/{run_name}", device="cuda:0")
itda(x, x)
#%%
