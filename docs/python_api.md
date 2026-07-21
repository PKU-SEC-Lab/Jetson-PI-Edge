# Python Foreground API

`jetson_pi_foreground` is a NumPy client for the persistent llama-server
`/foreground/*` API. The server owns the model, CUDA context, graph cache,
and foreground session. Each `predict()` call returns a contiguous float32
NumPy action array together with the full server response.

Use `ForegroundSession` to connect to a server that is already running:

```python
import numpy as np

from jetson_pi_foreground import ForegroundSession

session = ForegroundSession("http://127.0.0.1:8080", timeout=300)
state = np.zeros(32, dtype=np.float32)
action, metadata = session.predict(
    image_paths=["view0.jpg", "view1.jpg"],
    prompt="/do something",
    state=state,
    reset=True,
)
```

Use `ManagedForegroundSession` when Python should start and stop the server:

```python
from jetson_pi_foreground import ManagedForegroundSession

session = ManagedForegroundSession(
    server_path="/path/to/build/bin/llama-server",
    model_path="/path/to/pi0-model.gguf",
    mmproj_path="/path/to/mmproj-model.gguf",
    gpu=0,
    port=8080,
    timeout=300,
)

try:
    action, metadata = session.predict(
        image_paths=image_paths,
        prompt=prompt,
        state=state,
        reset=True,
    )
finally:
    session.close()
```

Keep the same session alive for repeated control steps. This keeps the model
and CUDA context loaded and allows compatible computation graphs to be reused.
Use `reset=True` for the first step or when beginning a new foreground session.

The managed client requires NumPy. It starts the existing `llama-server`
executable as a child process and does not require pybind11.
