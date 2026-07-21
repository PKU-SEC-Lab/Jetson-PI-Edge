# Contributing to Jetson-PI-Edge

Jetson-PI-Edge is a real-time inference engine for PI0 and PI0.5 policies.
Contributions are welcome, but changes must preserve predictable latency,
explicit backend selection, stable public APIs, numerical correctness, and
clear failure modes.

## Start Here

Before opening a pull request:

1. Read the relevant documentation:
   - Build: [`docs/build.md`](docs/build.md)
   - Model preparation: [`docs/model_conversion.md`](docs/model_conversion.md)
   - Foreground server: [`docs/foreground_server_usage.md`](docs/foreground_server_usage.md)
   - Server development: [`tools/server/README-dev.md`](tools/server/README-dev.md)
2. Search existing issues and pull requests to avoid duplicate work.
3. Build the affected targets locally.
4. Run the smallest test set that covers the change, then broaden testing
   according to risk.
5. For runtime changes, report the exact hardware, backend, build command,
   model, latency measurement method, and numerical comparison.

## Repository Rules

### Scope

- Keep each change focused on one feature, fix, or model path.
- Reuse existing llama.cpp, mtmd, and Jetson-PI infrastructure before adding
  a new subsystem.
- Keep model-specific behavior out of generic llama.cpp code when a
  Jetson-PI-specific layer can own it.
- Large or cross-cutting features should be discussed in an issue before the
  implementation is finalized.

### Public APIs

The following surfaces are public contracts:

- The `jetson_pi_llm`, `jetson_pi_mllm`, and `jetson_pi_pi0` C APIs.
- The documented `/foreground/*` HTTP endpoints.
- The optional Jetson-PI Python API.

Do not remove or silently change documented signatures, tensor layouts,
dtypes, state behavior, or output semantics. Additive changes are preferred.
Update the corresponding header, user documentation, and tests in the same
change.

### Backend Routing

- Backend selection must be explicit. Do not silently fall back from a
  requested accelerator to CPU.
- Unsupported backends, shapes, layouts, and model combinations must fail
  with a clear error.
- Backend-specific code must be guarded by matching build-time and runtime
  checks.
- A change for one backend must not alter the default behavior of unrelated
  backends.

### Python Bindings and CMake Ownership

Every Python binding must have matching CMake target ownership:

- A symbol exported unconditionally by pybind11 must be linked in every build
  configuration that produces the module.
- Optional implementations and their bindings must use the same compile-time
  guard, or the binding must raise a clear `not built` or `not supported`
  exception.
- Python bindings must be opt-in and must not add a Python or pybind11
  dependency to the default C++ build.
- Validate both the extension build and Python import. A successful linker
  invocation is not enough because missing symbols may appear at import time.

For NumPy inputs, document and validate dtype, rank, shape, memory layout, and
ownership. Do not silently reinterpret NCHW as NHWC, change element types, or
retain pointers to temporary arrays.

### Error Handling

Do not continue after model loading, backend, allocation, graph execution, or
shape validation errors. Public APIs must return or raise an error that names
the failed operation and includes relevant dimensions or configuration when
available. Undefined outputs, all-zero fallbacks, and warning-only failures
are not acceptable.

### Runtime and Graph Reuse

- Model and backend state belongs to the inference object that created it.
- Hot inference paths should avoid model reloads and avoid unnecessary memory
  allocation.
- Graph reuse must preserve numerical behavior. A fallback that rebuilds a
  compatible graph must be observable in performance measurements rather than
  hidden behind an unsupported performance claim.
- Request state, KV state, graph state, and model lifetime are separate
  contracts. Document which of them persist across calls.
- A single non-thread-safe inference handle must serialize concurrent calls or
  reject them clearly.

### Numerical and Performance Validation

Use the correct metric for each claim:

- End-to-end API latency includes Python conversion, preprocessing, inference,
  and output construction.
- Component latency may report ViT, language encoder, action expert, or graph
  execution independently.
- Do not compare component timing directly with end-to-end timing.

For deterministic tests, compare outputs exactly when practical. For
floating-point PI action outputs, report maximum error and cosine similarity
against the unchanged reference path. Performance-sensitive changes must
include warmup, sample count, P50 latency, and the graph-reuse configuration.

## Development and Testing

Use an out-of-tree build directory:

```bash
cmake -S . -B build
cmake --build build -j
```

Run focused tests for the touched component. Changes to core llama.cpp or ggml
must also run the relevant upstream tests. Changes to a ggml operator require
`test-backend-ops` on the affected backends.

Python binding changes must at least verify:

```bash
cmake -S . -B build-python \
  -DJETSON_PI_BUILD_PYTHON=ON \
  -DJETSON_PI_BUILD_MTMD=ON
cmake --build build-python -j --target jetson_pi

python -c "import jetson_pi; print(jetson_pi.__doc__)"
```

Run model-backed integration tests on each affected model/backend combination
when checkpoints and hardware are available. If a required fixture or device
is unavailable, state that explicitly and report the tests that were run.

## Coding Guidelines

- Follow the style of surrounding code.
- Use 4 spaces for indentation and keep braces on the same line.
- Use `snake_case` for C, C++, and Python names.
- Use lowercase dash-separated C/C++ filenames and lowercase underscore-
  separated Python filenames.
- Prefer sized integer types in public C APIs.
- Keep comments concise and explain non-obvious invariants rather than
  restating the code.
- Avoid unnecessary templates, dependencies, headers, and generated files.
- Use `clang-format` version 15 or newer for C/C++ changes when needed.
- Preserve compatibility across supported operating systems, architectures,
  and build configurations.

## Pull Request Checklist

Before requesting review:

- Rebase or fast-forward onto the latest `master`.
- Keep the change scoped to one behavior.
- Update documentation for user-visible API or build changes.
- Add or update tests.
- Include exact build and validation commands and their results.
- Include hardware, driver, toolkit, backend, model, and timing details for
  runtime changes.
- Verify that default builds remain unchanged when an optional feature is
  disabled.
- Do not commit build outputs, model files, local logs, caches, or credentials.
- Mention unsupported hardware and unavailable fixtures explicitly.

Expect review requests concerning correctness, performance, portability, and
long-term maintenance. Contributors are expected to understand and maintain
the code they submit.

## Commit Style

Use short, direct, technical commit subjects, for example:

```text
python: add NumPy PI policy inference binding
mtmd: fix PI0 image shape validation
server: preserve foreground action dimensions
```

Prefer small commits whose tests and benchmarks map directly to the changed
behavior.
