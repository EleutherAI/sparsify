#%%
%load_ext autoreload
%autoreload 2
from sparsify.__main__ import load_artifacts, RunConfig
from sparsify.sparse_coder import SparseCoder

hook = "layers.10.mlp"
device = "cuda"
#%%
# name = "sliced-norm_encoder-muon"
# name = "trans_32"
name = "encoder_normed_highlr"
sparse_coder = SparseCoder.load_from_disk(f"checkpoints/{name}/{hook}", device=device)
#%%
conf = RunConfig(sae=sparse_coder, dataset="NeelNanda/pile-10k")
model, dataset = load_artifacts(conf, 0)
# %%
import torch
torch.set_grad_enabled(False)
model.to(device)
#%%
#%%
from collections import OrderedDict
for m in model.modules():
    m._forward_hooks = OrderedDict()
ii = dataset["input_ids"][:32, :128].cuda()
def fn(m, i, o):
    if isinstance(i, tuple):
        i = i[0]
    if isinstance(o, tuple):
        o = o[0]
    i = i[:, 1:]
    o = o[:, 1:]
    i = i.reshape(-1, i.shape[-1])
    o = o.reshape(-1, o.shape[-1])
    # sparse_coder.cfg.ito = True
    result = sparse_coder(i, o)
    sparse_coder.cfg.ito = False    
    print(result.fvu)
with torch.no_grad():
    model.get_submodule(hook).register_forward_hook(fn)
    model(ii);
# %%