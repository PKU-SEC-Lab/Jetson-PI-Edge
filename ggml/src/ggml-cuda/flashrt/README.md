# FlashRT layer for ggml-cuda (Jetson AGX Thor, SM110)

Opt-in integration of FlashRT's NVFP4 kernels and fused-structure windows
into the CUDA backend, for pi0/pi0.5 on Jetson AGX Thor. Enabled with
`-DGGML_CUDA_FLASHRT=ON`; without the flag the build is stock llama.cpp.

Kernel sources are consumed from the `flashrt-public` submodule in this
directory (override with `-DGGML_CUDA_FLASHRT_PUBLIC_DIR=<checkout>`).
The adapter compiles as a separate object library with `-arch=sm_110a`
and only C symbols cross into ggml-cuda; shared kernels and dispatch are
untouched.

Full documentation lives with the adapter, in the submodule under
`flash_rt/structures/adapters/ggml/`:

- `README.md` — architecture and measured performance
- `USAGE.md` — build options, model preparation, runtime switches
- `TESTING.md` — operator tests, qualification gates, benchmark and
  parity methodology
- `DEVELOPMENT.md` — how the layer is organized and extended
