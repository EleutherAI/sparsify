from dataclasses import replace

import torch
from torch import Tensor

from .sparse_coder import MidDecoder, SparseCoder


class CrossLayerRunner(object):
    def __init__(self):
        self.outputs = {}
        self.to_restore = []

    def encode(self, x: Tensor, sparse_coder: SparseCoder, **kwargs):
        out_mid = sparse_coder(
            x=x,
            y=None,
            return_mid_decoder=True,
            **kwargs,
        )
        return out_mid

    def decode(
        self,
        mid_out: MidDecoder,
        y: Tensor,
        module_name: str,
        detach_grad: bool = False,
    ):
        self.outputs[module_name] = mid_out

        candidate_indices = []
        candidate_values = []
        hookpoints = []
        layer_mids = []
        output = 0
        to_delete = set()
        out, hookpoint = None, None
        if detach_grad:
            self.to_restore.clear()
        for i, (hookpoint, layer_mid) in enumerate(self.outputs.items()):
            if detach_grad:
                layer_mid.detach()
            if layer_mid.sparse_coder.cfg.divide_cross_layer:
                divide_by = max(1, len(self.outputs) - 1)
            else:
                divide_by = 1
            layer_mids.append(layer_mid)
            hookpoints.append(hookpoint)
            candidate_indices.append(
                layer_mid.latent_indices + i * layer_mid.sparse_coder.num_latents
            )
            candidate_values.append(layer_mid.current_latent_acts)
            if detach_grad:
                self.to_restore.append((layer_mid, layer_mid.will_be_last))
            if layer_mid.will_be_last:
                to_delete.add(hookpoint)
            if not mid_out.sparse_coder.cfg.do_coalesce_topk:
                out = layer_mid(
                    y,
                    addition=(0 if hookpoint != module_name else (output / divide_by)),
                    no_extras=hookpoint != module_name,
                    denormalize=hookpoint == module_name,
                )
                if hookpoint != module_name:
                    output += out.sae_out
            else:
                layer_mid.next()

        if mid_out.sparse_coder.cfg.do_coalesce_topk:
            candidate_indices = torch.cat(candidate_indices, dim=1)
            candidate_values = torch.cat(candidate_values, dim=1)
            best_values, best_indices = torch.topk(
                candidate_values, k=mid_out.sparse_coder.cfg.k, dim=1
            )
            best_indices = torch.gather(candidate_indices, 1, best_indices)
            if mid_out.sparse_coder.cfg.coalesce_topk == "concat":
                best_indices = best_indices % mid_out.sparse_coder.num_latents
                new_mid_out = mid_out.copy(
                    indices=best_indices,
                    activations=best_values,
                )
                out = new_mid_out(y, index=0, add_post_enc=False)
                del mid_out.x
            elif mid_out.sparse_coder.cfg.coalesce_topk == "per-layer":
                for i, layer_mid in enumerate(layer_mids):
                    hookpoint = hookpoints[i]
                    is_ours = hookpoint == module_name
                    if not is_ours:
                        continue
                    num_latents = layer_mid.sparse_coder.num_latents
                    if is_ours:
                        best_indices_local = best_indices
                        best_values_local = best_values
                    else:
                        best_indices_local = None
                        best_values_local = None
                    new_mid_out = layer_mid.copy(
                        indices=best_indices_local,
                        activations=best_values_local,
                    )
                    out = new_mid_out(
                        y,
                        layer_mid.index - 1,
                        add_post_enc=False,
                        addition=(0 if hookpoint != module_name else output),
                        no_extras=hookpoint != module_name,
                        denormalize=hookpoint == module_name,
                    )
                    if hookpoint != module_name:
                        output += out.sae_out
                    else:
                        out = replace(
                            out,
                            latent_indices=(out.latent_indices % num_latents)
                            * (out.latent_indices // num_latents == i),
                        )
            else:
                raise ValueError("Not implemented")

        # last output guaranteed to be the current layer
        assert hookpoint == module_name

        for hookpoint in to_delete:
            del self.outputs[hookpoint]
        return out

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        sparse_coder: SparseCoder,
        module_name: str,
        detach_grad: bool = False,
        **kwargs,
    ):
        mid_out = self.encode(x, sparse_coder, **kwargs)
        return self.decode(mid_out, y, module_name, detach_grad)

    def restore(self):
        for restorable, was_last in self.to_restore:
            if was_last:
                restorable.restore(True)

    def reset(self):
        self.outputs.clear()
        self.to_restore.clear()
