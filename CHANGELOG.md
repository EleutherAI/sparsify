# CHANGELOG


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
