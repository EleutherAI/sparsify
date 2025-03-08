#%%
%env CUDA_VISIBLE_DEVICES=6
from transformers import AutoTokenizer, AutoModelForCausalLM

model_names = dict(
    monet_vd_s = "MonetLLM/monet-vd-850M-100BT-hf",
    # monet_vd_l = "MonetLLM/monet-vd-1.4B-100BT-hf",
    # monet_hd_s = "MonetLLM/monet-hd-850M-100BT-hf",
    # monet_hd_l = "MonetLLM/monet-hd-1.4B-100BT-hf",
    # llama_1_4b = "meta-llama/Llama-3.2-1B",
    # llama_1_1b = "TinyLlama/TinyLlama-1.1B-Chat-v0.1",
    
    # pythia_410m = "EleutherAI/pythia-410m"
)
tokenizers = {name: AutoTokenizer.from_pretrained(model) for name, model in model_names.items()}
#%%
import torch
torch.set_grad_enabled(False)
models = {name: AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.bfloat16,).cuda() for name, model in model_names.items()}
#%%
from itertools import product
import torch.utils.benchmark
from torch.utils.flop_counter import FlopCounterMode

def time_compiled(model, input_ids):
    flop_counter = FlopCounterMode()
    with flop_counter:
        model(input_ids)
    total_flops = flop_counter.get_total_flops()
    model = torch.compile(model)
    model(input_ids)
    timer = torch.utils.benchmark.Timer(
        stmt="model(input_ids)",
        globals={"model": model, "input_ids": input_ids},
        setup="model(input_ids)",
    ).blocked_autorange()
    return timer.mean, total_flops

# model_results = []
# for (batch_size, sequence_length) in product((1, 4, 16), (64, 128, 256, 1024)):
#     for model_name, model in models.items():
#         input_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long, device="cuda")
#         time, flops = time_compiled(model, input_ids)
#         model_results.append((model_name, batch_size, sequence_length, time, flops))
# #%%
# import pandas as pd
# df = pd.DataFrame(model_results, columns=['model', 'batch_size', 'sequence_length', 'time', "flops"])
# df.to_csv("monet_vs_transformer.csv", index=False)
#%%
models["monet_vd_s"].config
#%%
input_ids = torch.randint(0, models["monet_vd_s"].config.vocab_size,
                          (256, 128), dtype=torch.long, device="cuda")
for m in models["monet_vd_s"].model.modules():
    m._forward_hooks.clear()
saved = {}
models["monet_vd_s"].model.layers[8].moe.register_forward_hook(lambda m, i, o: saved.setdefault("moe", (i, o)) and None)
models["monet_vd_s"](input_ids)
for m in models["monet_vd_s"].model.modules():
    m._forward_hooks.clear()
#%%
x, g1, g2 = saved["moe"][0]
moe = models["monet_vd_s"].model.layers[9].moe
moe(x, g1, g2).shape
#%%
torch.set_grad_enabled(False)


def timing(name, fn, *args):
    # fn = torch.compile(fn)
    fn(*args)
    timer = torch.utils.benchmark.Timer(
        stmt="fn(*args)",
        globals={"fn": fn, "args": args},
        setup="fn(*args)",
    ).blocked_autorange()
    print(name, timer.mean)

@torch.compile
def vanilla_forward(
    self, x: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor
) -> torch.Tensor:
    g1, g2 = g1.type_as(x), g2.type_as(x)
    x1 = self.act_fn(self.u1(x).unflatten(-1, (self.config.moe_experts, -1)))
    x2 = self.act_fn(self.u2(x).unflatten(-1, (self.config.moe_experts, -1)))

    x11 = self.v11(torch.einsum("btim,bthi->btim", x1, g1).flatten(-2))
    # x12 = self.v12(torch.einsum("btjm,bthj,bthi->btim", x2, g2, g1).flatten(-2))
    x12 = x11 * 0
    x13 = torch.einsum("bthi,id->btd", g1, self.b1.type_as(x))

    x22 = self.v22(torch.einsum("btjm,bthj->btjm", x2, g2).flatten(-2))
    # x21 = self.v21(torch.einsum("btim,bthi,bthj->btjm", x1, g1, g2).flatten(-2))
    x21 = x22 * 0 
    x23 = torch.einsum("bthj,jd->btd", g2, self.b2.type_as(x))

    return torch.cat((x11 + x12 + x13, x21 + x22 + x23), dim=-1)

from sparsify.kernels import triton_sparse_dense_matmul

@torch.compile
def topk_forward(
    self, x: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor,
    v11, v22
):
    k = 16
    g1, g2 = g1.type_as(x), g2.type_as(x)
    g1_full_v, g1_full_i = g1.sum(-2).topk(k, dim=-1)
    g2_full_v, g2_full_i = g2.sum(-2).topk(k, dim=-1)
    g1_v, g1_i = g1.topk(k, dim=-1)
    g2_v, g2_i = g2.topk(k, dim=-1)
    x1 = self.act_fn(self.u1(x).unflatten(-1, (self.config.moe_experts, -1)))
    x2 = self.act_fn(self.u2(x).unflatten(-1, (self.config.moe_experts, -1)))

    x13 = torch.einsum("bthi,id->btd", g1, self.b1.type_as(x))
    post_g1 = torch.einsum("btim,bthi->btim", x1, g1).flatten(-2)
    # x11 = self.v11(post_g1)
    gathered_post_g1 = torch.gather(post_g1, -1, g1_full_i)
    indices = g1_full_i.flatten(0, -2)
    weights = (gathered_post_g1 * g1_full_v).flatten(0, -2)
    x11 = triton_sparse_dense_matmul(
        indices, weights,
        # self.v11.weight.T.contiguous(),
        v11,
    )  # + self.v11.bias
    x11 = x11.reshape(*g1_full_i.shape[:-1], -1)
    # x12 = self.v12(torch.einsum("btjm,bthj,bthi->btim", x2, g2, g1).flatten(-2))
    x12 = x11 * 0

    x23 = torch.einsum("bthj,jd->btd", g2, self.b2.type_as(x))
    post_g2 = torch.einsum("btjm,bthj->btjm", x2, g2).flatten(-2)
    gathered_post_g2 = torch.gather(post_g2, -1, g2_full_i)
    indices = g2_full_i.flatten(0, -2)
    weights = (gathered_post_g2 * g2_full_v).flatten(0, -2)
    x22 = triton_sparse_dense_matmul(
        indices, weights,
        v22,
        # self.v22.weight.T.contiguous(),
    )
    x22 = x22.reshape(*g2_full_i.shape[:-1], -1)
    # x22 = self.v22(post_g2)
    # x21 = self.v21(torch.einsum("btim,bthi,bthj->btjm", x1, g1, g2).flatten(-2))
    x21 = x22 * 0

    return torch.cat((x11 + x12 + x13, x21 + x22 + x23), dim=-1)

v11 = moe.v11.weight.T.contiguous()
v22 = moe.v22.weight.T.contiguous()
timing("topk_forward", topk_forward, moe, x, g1, g2, v11, v22)
timing("vanilla_forward", vanilla_forward, moe, x, g1, g2)
#%%
# g2.sort(dim=-1).values[..., -1].mean(), g1.shape
g2.topk(12, dim=-1).values.sum(-1).mean()
#%%
with torch.inference_mode():
    g = torch.einsum("...hi,...hj->...ij", g1, g2)
    g = g.reshape(g.shape[:-2] + (-1,))
    print(g.topk(512, dim=-1).values.sum(-1).mean())
#%%
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_theme()
m_v_t = pd.read_csv("monet_vs_transformer.csv")
grouped = m_v_t.groupby(["model"])
display_name = dict(
    llama_1_1b = "TinyLlama-1.1B",
    llama_1_4b = "Llama-3.2-1B",
    monet_vd_s = "Monet-850M",
    monet_vd_l = "Monet-1.4B",
    monet_hd_s = "MonetHD-850M",
    monet_hd_l = "MonetHD-1.4B",
    pythia_410m = "Pythia-410M"
)
plt.figure(figsize=(8, 6))
for (model_name,), model_data in sorted(grouped):
    # draw Pareto frontier
    model_data = model_data.sort_values("flops")
    x, y = model_data["flops"], model_data["time"]
    x_greatest_so_far = np.maximum.accumulate(x)
    y_greatest_so_far = np.maximum.accumulate(y)
    # plt.plot(x_greatest_so_far, y_greatest_so_far, label=None)
    plt.scatter(x, y, 
                label=display_name[model_name], marker="x" if "monet" in model_name else "o")
    
    # plt.plot(x, y, label=None)
plt.xlabel("FLOPs")
plt.ylabel("Time (s)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("monet_vs_transformer.svg")
plt.show()
    
# %%
from sparsify.monet import MonetConfig, MonetMoVDE
