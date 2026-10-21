import torch

from sparsify import SparseCoder, SparseCoderConfig


def test_multi_topk_fvu_includes_skip_connection():
    """The Multi-TopK reconstruction must include the skip connection, like the
    main TopK reconstruction does; otherwise the 4k latents are trained to also
    reconstruct the part of the target that ``W_skip`` already predicts.

    This runs on CPU using the eager decoder fallback, so it requires no GPU.
    """
    torch.manual_seed(0)

    d_in = 16
    sae = SparseCoder(
        d_in,
        SparseCoderConfig(num_latents=64, k=4, multi_topk=True, skip_connection=True),
    )
    x = torch.randn(32, d_in)

    # The skip connection alone reconstructs the input exactly.
    assert sae.W_skip is not None and sae.W_dec is not None
    with torch.no_grad():
        sae.W_skip.copy_(torch.eye(d_in))
        sae.W_dec.zero_()
        sae.b_dec.zero_()
    out = sae(x)
    torch.testing.assert_close(out.fvu, torch.tensor(0.0))
    torch.testing.assert_close(out.multi_topk_fvu, torch.tensor(0.0))

    # In general, Multi-TopK FVU is the FVU of the 4k-latent decode plus the skip term.
    torch.nn.init.normal_(sae.W_skip, std=0.1)
    torch.nn.init.normal_(sae.W_dec)
    out = sae(x)
    _, _, pre_acts = sae.encode(x)
    top_acts, top_indices = pre_acts.topk(4 * sae.cfg.k, sorted=False)
    sae_out = sae.decode(top_acts, top_indices) + x @ sae.W_skip.mT
    expected = (sae_out - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
    torch.testing.assert_close(out.multi_topk_fvu, expected)
