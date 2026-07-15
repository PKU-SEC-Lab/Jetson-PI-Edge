# Foreground Server Usage

The foreground server exposes a persistent HTTP session for PI0 and PI0.5 robot action inference. Inputs are submitted in stages: images first, then robot state, then instruction text. The final `/foreground/infer` request consumes the pending inputs and returns action tensors plus timing information.

## Session Model

The server keeps a foreground session with:

- `pending_text`: text and media markers waiting for the next inference request.
- `pending_bitmaps`: images waiting to be consumed.
- `history_size`: foreground conversation history length.
- `n_past`: current context position.
- `state_flag`: whether a PI state vector has been loaded.

After a successful `/foreground/infer` call, pending images and pending text are consumed. The foreground history and `n_past` remain until `/foreground/reset` is called.

## Start The Server

```bash
PI_MODEL=auto \
PI0_ACTION_NOISE_BIN=/path/to/noise.bin \
./build/bin/llama-server \
  -m /path/to/pi_llm.gguf \
  --mmproj /path/to/mmproj.gguf \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

`PI_MODEL=auto` reads GGUF metadata and PI tensor names. Use `PI_MODEL=pi0` or `PI_MODEL=pi05` when you want to force a specific branch.

## Minimal End-To-End Example

Reset the session:

```bash
curl -X POST http://127.0.0.1:8080/foreground/reset
```

Submit images:

```bash
curl -X POST http://127.0.0.1:8080/foreground/image \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/image_1.png"}'

curl -X POST http://127.0.0.1:8080/foreground/image \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/image_2.png"}'
```

Submit state:

```bash
curl -X PUT http://127.0.0.1:8080/foreground/state \
  -H 'Content-Type: application/json' \
  -d '{"state":"1.8731,-1.0370,1.9652,7.0876,0.2546,-9.1432,-0.0147,-0.5037,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"}'
```

Run inference:

```bash
curl -X POST http://127.0.0.1:8080/foreground/infer \
  -H 'Content-Type: application/json' \
  -d '{"text":"pick up the object and place it into the tray"}'
```

Inspect the current session:

```bash
curl http://127.0.0.1:8080/foreground/session
```

## Endpoints

### `GET /foreground/session`

Returns the current foreground session state.

Typical fields:

- `pending_text`
- `pending_bitmaps`
- `history_size`
- `n_past`
- `state_flag`
- `pi_model`

### `POST /foreground/image`

Adds one image to the current session.

Request:

```json
{
  "path": "/abs/path/to/image.png"
}
```

The server loads the image, appends one media marker to pending text, and increments `pending_bitmaps`.

### `PUT /foreground/state`

Loads the robot state vector for the current session.

Request:

```json
{
  "state": "1.0,2.0,3.0,..."
}
```

The state string is parsed as comma-separated floats and passed into the PI context with `llama_set_pi0_state`.

### `POST /foreground/infer`

Consumes pending images, the latest loaded state, and the request text.

Request:

```json
{
  "text": "pick up the object and place it into the tray"
}
```

For non-PI paths, `n_predict` may also be supplied. For PI action models, the response is action-oriented and does not depend on token generation length.

### `POST /foreground/reset`

Clears the foreground session:

- pending images
- pending text
- foreground history
- context position
- `state_flag`
- cached PI result buffers

## PI Response Fields

`POST /foreground/infer` can return:

- `content`: text content, usually empty for PI action inference.
- `is_pi0`: whether the PI action path was used.
- `pi_model`: active PI model family.
- `n_past`: context position after inference.
- `history_size`: foreground history length.
- `state`: state vector used or returned by the PI path.
- `action`: flattened internal action buffer.
- `action_final`: final flattened action output.
- `encode_ms`: encode latency.
- `decode_ms`: action expert latency.
- `total_ms`: model-side total latency.
- `batch_build_ms`, `output_extract_ms`, `batch_free_ms`: lower-level runtime timing fields.
- `timing_breakdown_ms`: HTTP handler and evaluation timing details.

## Operational Notes

- The number of media markers must match the number of pending bitmaps.
- Pending images are consumed after a successful inference request.
- Submit new images before each new foreground inference round.
- PI0 and PI0.5 support up to three foreground images in the current action path.
- `state_flag` remains set until `/foreground/reset`.
- If auto detection is not desired, set `PI_MODEL=pi0` or `PI_MODEL=pi05` explicitly.
