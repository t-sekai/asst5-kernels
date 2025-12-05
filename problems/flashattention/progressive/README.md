# Progressive FlashAttention Kernels

This directory contains three incremental versions of the CUDA kernel so you can
adopt improvements one at a time or benchmark them independently. Each file is
self-contained (it exposes its own `flash_attention_forward_step*` entry point)
so you can swap it into `submission.py` when you want to test that version.

| File | Focus | Key Changes vs. baseline |
| --- | --- | --- |
| `step1_block_parallel.cu` | **Thread-block parallelism** | moves from one-thread blocks to 128-thread blocks, caches the query row in shared memory, keeps the running output vector in SRAM, and uses warp-level reductions for the dot product. |
| `step2_shared_tile.cu` | **HBM traffic reduction** | builds on Step 1 and stages blocks of K/V rows (32 at a time) into shared memory so each tile is fetched once, reusing the staged values for all threads in the block. |
| `step3_vectorized_shared.cu` | **Vectorized math on SRAM** | builds on Step 2 and processes four elements per instruction (`float4` loads/stores inside the shared-memory buffers) to raise FMA throughput and reduce shared-memory transactions. |

### Using a specific step
1. Pick the desired file in this directory.
2. Replace the CUDA string that `submission.py` injects (or point your build at
   the chosen `.cu` file) and call the matching function
   (`flash_attention_forward_step1/2/3`).
3. Test/benchmark as usual.

The progressive layout should make it easier to study the impact of each
optimization independently or to mix ideas across steps.
