import torch

from sparsify import SparseCoder, SparseCoderConfig
from sparsify.utils import decoder_impl


def test_auxk_loss_does_not_double_count_b_dec():
    """The AuxK loss target ``e = y - sae_out`` already accounts for ``b_dec``
    (since ``sae_out`` includes it), so the second decoder pass used to compute
    ``e_hat`` must *not* add ``b_dec`` again. See issue #132.

    This runs on CPU using the eager decoder fallback, so it requires no GPU.
    """
    torch.manual_seed(0)

    d_in = 16
    num_latents = 32
    k = 4
    batch = 8

    sae = SparseCoder(
        d_in,
        SparseCoderConfig(num_latents=num_latents, k=k),
    )

    # Give b_dec a nonzero value; this is the normal case after init centers it
    # on the data mean, and is exactly the situation the bug affects.
    with torch.no_grad():
        sae.b_dec.copy_(torch.randn(d_in))

    x = torch.randn(batch, d_in)
    dead_mask = torch.ones(num_latents, dtype=torch.bool)

    out = sae(x, dead_mask=dead_mask)

    # Recompute the AuxK loss by hand, decoding *without* re-adding b_dec.
    top_acts, top_indices, pre_acts = sae.encode(x)
    sae_out = sae.decode(top_acts, top_indices)
    e = x - sae_out
    total_variance = (x - x.mean(0)).pow(2).sum()

    num_dead = int(dead_mask.sum())
    k_aux = x.shape[-1] // 2
    scale = min(num_dead / k_aux, 1.0)
    k_aux = min(k_aux, num_dead)

    auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
    auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

    # Correct target: decode without adding b_dec a second time.
    assert sae.W_dec is not None
    e_hat = decoder_impl(auxk_indices, auxk_acts.to(sae.dtype), sae.W_dec.mT)
    expected_auxk_loss = scale * (e_hat - e.detach()).pow(2).sum() / total_variance

    torch.testing.assert_close(out.auxk_loss, expected_auxk_loss)

    # Sanity check: the buggy formulation (re-adding b_dec) gives a *different*
    # value when b_dec != 0, so this test would actually fail without the fix.
    buggy_e_hat = e_hat + sae.b_dec
    buggy_auxk_loss = scale * (buggy_e_hat - e.detach()).pow(2).sum() / total_variance
    assert not torch.allclose(out.auxk_loss, buggy_auxk_loss)
