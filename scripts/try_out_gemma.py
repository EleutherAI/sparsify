# %%
from IPython import get_ipython

if (ip := get_ipython()) is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-2-2b"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %%

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

if 1:
    # https://huggingface.co/mntss/skip-transcoders-gemma-2-2b/resolve/main/layer_17.safetensors
    params_path = hf_hub_download(
        repo_id="mntss/skip-transcoders-gemma-2-2b", filename="layer_17.safetensors"
    )
    params = load_file(params_path, device=device)
    # print({k: v.shape for k, v in params.items()})
    W_enc = params["W_enc"]
    W_dec = params["W_dec"].T
    W_skip = params["W_skip"]
    b_dec = params["b_dec"]
    b_enc = params["b_enc"]
    d_in = b_dec.shape[0]
    d_f = b_enc.shape[0]
else:
    params_path = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-transcoders",
        filename="layer_17/width_16k/average_l0_112/params.npz",
    )
    layer_idx = 17
    params_npz = np.load(params_path)
    {k: v.shape for k, v in params_npz.items()}
    W_enc = params_npz["W_enc"]
    W_dec = params_npz["W_dec"]
    b_dec = params_npz["b_dec"]
    b_enc = params_npz["b_enc"]
    threshold = params_npz["threshold"]

    b_enc_orig = b_enc
    b_enc, b_enc_post = b_enc - threshold, threshold

    W_enc = torch.from_numpy(W_enc.T).to(device)
    W_dec = torch.from_numpy(W_dec).to(device)
    b_dec = torch.from_numpy(b_dec).to(device)
    b_enc = torch.from_numpy(b_enc).to(device)
    b_enc_post = torch.from_numpy(b_enc_post).to(device)

    b_enc_orig = torch.from_numpy(b_enc_orig).to(device)
    threshold = b_enc_post

d_in = b_dec.shape[0]
d_f = b_enc.shape[0]

pre_mlp_rmsnorm = 1.0 + model.model.layers[layer_idx].pre_feedforward_layernorm.weight
post_mlp_rmsnorm = 1.0 + model.model.layers[layer_idx].post_feedforward_layernorm.weight

W_enc = W_enc / pre_mlp_rmsnorm
W_skip = W_skip / pre_mlp_rmsnorm
W_dec = W_dec / post_mlp_rmsnorm
b_dec = b_dec / post_mlp_rmsnorm
W_skip = W_skip / post_mlp_rmsnorm[:, None]
# %%
from sparsify.config import SparseCoderConfig
from sparsify.sparse_coder import SparseCoder

config = SparseCoderConfig(
    activation="topk",
    num_latents=d_f,
    k=64,
    transcode=True,
    skip_connection=True,
)
coder = SparseCoder(d_in, config, device=device)
coder.encoder.weight.data[:] = W_enc
coder.encoder.bias.data[:] = b_enc
# coder.post_enc.data[:] = b_enc_post
coder.W_dec.data[:] = W_dec
coder.b_dec.data[:] = b_dec
coder.W_skip.data[:] = W_skip
##%%
from collections import OrderedDict

for m in model.modules():
    m._forward_hooks = OrderedDict()


def forward_hook(m, input, output):
    if isinstance(input, tuple):
        input = input[0]
    if isinstance(output, tuple):
        output = output[0]
    input = input[:, 1:]
    output = output[:, 1:]
    if len(input.shape) > 2:
        input = input.flatten(0, -2)
        output = output.flatten(0, -2)

    # output = model.model.layers[layer_idx].post_feedforward_layernorm(output)
    # output = output * (1 + model.model.layers[layer_idx].post_feedforward_layernorm.weight)
    # input = input / pre_mlp_rmsnorm
    # input = input / (d_in ** 0.5)
    # output = output / (d_in ** 0.5)
    out = coder(input, output)
    # print(out.fvu)

    res = out.sae_out
    # res = res / (1 + model.model.layers[layer_idx].post_feedforward_layernorm.weight)
    # res = res / post_mlp_rmsnorm

    print(torch.nn.functional.cosine_similarity(res, output).mean())
    # output[:, 1:] = res[:, 1:]
    # return output


model.model.layers[layer_idx].mlp.register_forward_hook(forward_hook)
text = "Hello, how are you?"
tokens = tokenizer(text, return_tensors="pt").to(device)
tokens = tokens.input_ids
model.eval()
with torch.no_grad():
    model(tokens)
    # out = model.generate(tokens, max_new_tokens=100)
# print(tokenizer.decode(out[0]))
# %%
import transformer_lens  # pip install transformer-lens

model_tl = transformer_lens.HookedTransformer.from_pretrained(
    "google/gemma-2-2b",
    # In Gemma 2, only the pre-MLP, pre-attention and final RMSNorms can
    # be folded in (post-attention and post-MLP RMSNorms cannot be folded in):
    fold_ln=True,
    # Only valid for models with LayerNorm, not RMSNorm:
    center_writing_weights=False,
    # These model use logits soft-capping, meaning we can’t center unembed:
    center_unembed=False,
)
# %%
model_tl.cfg.use_hook_mlp_in = True
# %%
with torch.no_grad():
    pre_resid = None

    def forward_hook_pre(resid, hook: transformer_lens.hook_points.HookPoint):
        global pre_resid
        pre_resid = resid

    def forward_hook(resid, hook: transformer_lens.hook_points.HookPoint):
        inputs = pre_resid[:, 1:].flatten(0, -2)
        outputs = resid[:, 1:].flatten(0, -2)
        # inputs = inputs / (d_in ** 0.5)
        # inputs = inputs * (d_in ** 0.5)
        outputs = outputs * (d_in**0.5)
        out = coder(inputs, outputs)
        print(out.fvu)
        # outputs = outputs * (1 + model.model.layers[layer_idx].post_feedforward_layernorm.weight)
        print(torch.nn.functional.cosine_similarity(outputs, out.sae_out).mean())

    model_tl.run_with_hooks(
        "Hello, how are you?",
        fwd_hooks=[
            (f"blocks.{layer_idx}.hook_mlp_in", forward_hook_pre),
            (f"blocks.{layer_idx}.hook_mlp_out", forward_hook),
        ],
    )
# %%
model_tl
