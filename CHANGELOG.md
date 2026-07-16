# CHANGELOG


## v1.3.3 (2026-07-16)

### Bug Fixes

- Don't double-count b_dec in TopK AuxK loss
  ([#132](https://github.com/EleutherAI/sparsify/pull/132),
  [`b1fbd0b`](https://github.com/EleutherAI/sparsify/commit/b1fbd0b2a119b7aced21e8ed0e8089e3ab2d7a94))

The AuxK loss target e = y - sae_out already accounts for b_dec, since sae_out is produced by
  decode() which adds b_dec. Computing e_hat via self.decode() added b_dec a second time, pulling
  dead latents toward (e - b_dec) and placing an unintended gradient on b_dec whenever b_dec != 0
  (the normal case after init centers it on the data mean).

Call decoder_impl directly so e_hat does not re-add b_dec.


## v1.3.2 (2026-07-16)

### Bug Fixes

- Make micro_acc_steps actually chunk activations
  ([`959030d`](https://github.com/EleutherAI/sparsify/commit/959030d1e4a15417efae9302f62da35ff34e654d))

`micro_acc_steps` has been a no-op since #60 (Support end-to-end training). That PR removed the loop
  that chunked the activations, but kept the `acc_steps = grad_acc_steps * micro_acc_steps`
  denominator that the loop existed to compensate for.

The result was worse than the flag simply being ignored: setting micro_acc_steps=N saved no memory
  at all, while still dividing the loss by an extra factor of N, silently scaling down the
  gradients.

Restore the chunking loop, so the existing `acc_steps`/`denom` factors are correct again. Chunking
  is only wired up for the `fvu` loss, where the backward pass happens inside the hook; the
  `ce`/`kl` losses are computed on the model's logits and need the full reconstruction in one piece,
  so combining them with micro_acc_steps > 1 now raises instead of being quietly ignored.

Note that chunking is exact only in the limit of large chunks: FVU normalizes by `total_variance`, a
  sum over the batch, computed per chunk against that chunk's own y.mean(0). Tiny chunks therefore
  diverge from an unchunked step (~22% of the update norm at 32 tokens/chunk, ~9% at 256),
  converging to float-noise exact by ~1024 tokens/chunk -- far below any realistic training config.
  The added test asserts this equivalence at 1024 tokens/chunk.

- Normalize micro_acc_steps chunks against unchunked batch variance
  ([`f19b274`](https://github.com/EleutherAI/sparsify/commit/f19b274ec3a84b43e67b03501e0dbbbf9d493d4e))

Each chunk was computing total_variance from its own local y.mean(0), which biased the
  FVU/auxk/multi-topk loss scale relative to an unchunked run (verified failing
  test_micro_acc_steps_matches_unchunked_update: ~3% off even at 1024 tokens/chunk, not shrinking
  with chunk size). SparseCoder.forward() now accepts an optional total_variance override; Trainer
  computes it once from the full pre-chunk batch and shares it across all chunks.

Also carries the embed_skip config/SparseCoder/Trainer plumbing this branch's own
  test_micro_acc_steps_with_embed_skip depends on, which wasn't committed here yet.

- Remove embed_skip from this branch, keep only the total_variance fix
  ([`001d98e`](https://github.com/EleutherAI/sparsify/commit/001d98eb1a0a71e1d605dc3c0d55aa53a5b662e5))

The previous commit accidentally bundled the embed_skip feature (an unrelated, separate piece of
  work) in with the total_variance fix. This branch should only contain the micro_acc_steps chunking
  fix plus the total_variance normalization fix on top of it.

- Shorten comment to satisfy ruff E501 line-length
  ([`4b35cee`](https://github.com/EleutherAI/sparsify/commit/4b35ceebfa4c1add37e6abde909b70d33f032273))


## v1.3.1 (2026-07-13)

### Bug Fixes

- Release
  ([`61b37db`](https://github.com/EleutherAI/sparsify/commit/61b37dba146ddfe79892d412e87190646ee0d1c8))


## v1.3.0 (2025-11-17)

### Features

- Pass arbitrary ds loading arguments
  ([`80ebed4`](https://github.com/EleutherAI/sparsify/commit/80ebed43f511601e73b71a9213d14a93e6ec686d))


## v1.2.2 (2025-09-30)

### Bug Fixes

- Revert "Merge pull request #118 from EleutherAI/exclude-follow-up"
  ([`304502c`](https://github.com/EleutherAI/sparsify/commit/304502c1a48b1aa6ae734fc3e8893d4743df6006))

This reverts commit 5c9c6fb89448feb4b87b23254a52ad02e60c10db, reversing changes made to
  bb792438e62b1ebe727720026fa7cd1d136752d1.


## v1.2.1 (2025-09-25)

### Bug Fixes

- Address code review comments - move import to top, restore TODO
  ([`f86219a`](https://github.com/EleutherAI/sparsify/commit/f86219a9a84f163c002159683c5917b11ea03c54))

- Auto-detect dtype from safetensors file to resolve loading mismatch
  ([`8fc921f`](https://github.com/EleutherAI/sparsify/commit/8fc921ffb8e938e78d954140a4caceaba952986c))


## v1.2.0 (2025-09-22)

### Features

- Exclude tokens defined by user from training
  ([`f346038`](https://github.com/EleutherAI/sparsify/commit/f3460385efc42fac6e760357f3c34562783b515c))


## v1.1.3 (2025-04-17)

### Bug Fixes

- Save best in dist mode
  ([`885bcd5`](https://github.com/EleutherAI/sparsify/commit/885bcd5c1e94d6b4d82b200545fe0ee1f830068e))


## v1.1.2 (2025-04-17)

### Bug Fixes

- Only drop in dist
  ([`b18bd0e`](https://github.com/EleutherAI/sparsify/commit/b18bd0e272044e42dce816583214e9d099484575))


## v1.1.1 (2025-04-16)

### Bug Fixes

- Hang when the number of examples is indivisible across processes
  ([`0f662ad`](https://github.com/EleutherAI/sparsify/commit/0f662adf7705a836aa8c910b39b33546a8cf7975))


## v1.1.0 (2025-04-16)

### Features

- Update from deprecated publish action
  ([`aafc9fc`](https://github.com/EleutherAI/sparsify/commit/aafc9fc049e7e8f18a017e99f122343f0bcb4006))

fix: deprecated initial release build


## v1.0.0 (2025-04-16)

### Features

- Empty commit for initial release
  ([`877ef6f`](https://github.com/EleutherAI/sparsify/commit/877ef6f7219e9a4424bf9cb51be5bef5ac2adca4))

BREAKING CHANGE: non-breaking inital major commit

### Breaking Changes

- Non-breaking inital major commit


## v0.0.1 (2025-04-16)

### Bug Fixes

- Change topk dim selection from 1 to -1
  ([`1cdb4a5`](https://github.com/EleutherAI/sparsify/commit/1cdb4a5bbe723b0ee0a0015f834d142e83facafe))

- Remove lint from CI, remove environment from CI, trigger release
  ([`f6dca80`](https://github.com/EleutherAI/sparsify/commit/f6dca80d4575fa1a58eac55cc7b24f802fa669db))
