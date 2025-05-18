from torch import Tensor

from .sparse_coder import SparseCoder


class CrossLayerRunner(object):
    def __init__(self):
        self.outputs = {}
        self.to_restore = []

    def run(
        self,
        x: Tensor,
        y: Tensor,
        sparse_coder: SparseCoder,
        module_name: str,
        detach_grad: bool = False,
        **kwargs,
    ):
        out_mid = sparse_coder(
            x=x,
            y=y,
            return_mid_decoder=True,
            **kwargs,
        )
        self.outputs[module_name] = out_mid

        output = 0
        to_delete = set()
        out, hookpoint = None, None
        if detach_grad:
            self.to_restore.clear()
        for hookpoint, layer_mid in self.outputs.items():
            if detach_grad:
                layer_mid.detach()
            if layer_mid.sparse_coder.cfg.divide_cross_layer:
                divide_by = max(1, len(self.outputs) - 1)
            else:
                divide_by = 1
            out = layer_mid(
                y,
                addition=(0 if hookpoint != module_name else (output / divide_by)),
                no_extras=hookpoint != module_name,
                denormalize=hookpoint == module_name,
            )
            if detach_grad:
                self.to_restore.append((layer_mid, out.is_last))
            if hookpoint == module_name:
                output = out.sae_out
            else:
                output += out.sae_out
            if out.is_last:
                to_delete.add(hookpoint)

        # last output guaranteed to be the current layer
        assert hookpoint == module_name

        for hookpoint in to_delete:
            del self.outputs[hookpoint]
        return out, output

    def restore(self):
        for restorable, was_last in self.to_restore:
            if was_last:
                restorable.restore(True)

    def reset(self):
        self.outputs.clear()
        self.to_restore.clear()
