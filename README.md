# orchestrator

One OpenAI-compatible endpoint, two swappable GPU slots, registry-driven model swaps.

A FastAPI proxy that fronts two inference slots on two GPUs, swaps models per-request from a registry, and enforces congestion-aware routing plus idle swap-back to a boot default.

---

## Read this first — what you have to bring

This repo contains **only the orchestrator**. It does not ship the inference stack it drives. A fresh clone will **not** serve a single token until you provide:

### Required

| What | Why |
|---|---|
| **NVIDIA GPUs + driver** (one per slot) | Two slots assume two GPUs. The systemd unit pre-flights `nvidia-smi -i 0` **and** `-i 1`. |
| **`llama-server` binary** — build [llama.cpp](https://github.com/ggml-org/llama.cpp) or install it | The orchestrator shells out to it. `ORCHESTRATOR_LLAMA_BIN` (default `/usr/local/bin/llama-server`) must point at a real binary or slot start fails. |
| **GGUF model weights** you actually want to serve | `registry.example.json` references 11 GGUFs by name. None are in the repo. Either download those exact files or edit the registry so every `gguf`/`mmproj` path points at models you have. |
| **`default_boot` in the registry** | Boot lineup of models for each slot. If the entries reference models that don't exist, the orchestrator comes up but slots go to `error`. |

### Optional — Strata backend

Needed only if any registry entry uses `"backend": "strata"`:

| What | Notes |
|---|---|
| [Strata](https://github.com/Niko1221/Strata) CLI + engine | Set `ORCHESTRATOR_STRATA_BIN` (default `/usr/local/bin/strata`). |
| A Strata profile JSON | Path comes from the entry's `strata_profile` field; example uses `$ORCHESTRATOR_DATA_DIR/profiles/strata-single.json`. Not shipped. |
| Strata model packs | Set `STRATA_DATA_DIR` in the service environment. |

### Optional — image-gen cooperation gate

The orchestrator returns a fast `503 [IMAGEGEN_ACTIVE]` while `$ORCHESTRATOR_MODE_STATE` contains `imagegen`, `imagegen-single`, `imagegen-single-fallback`, or `musicgen`. **The writer of that file is not in this repo.** If nothing ever writes it, `get_node_mode()` reads `unknown` and the gate silently never fires — everything else still works normally.

### Optional — page-cache warmup

If `$ORCHESTRATOR_WARM_CACHE_SCRIPT` points at a Python script, the orchestrator runs it in parallel with Strata VRAM drain to speed up Strata loads. Not shipped. Missing script = warning log, no failure.

---

## What IS in this repo

| Path | Purpose |
|---|---|
| `orchestrator.py` | The FastAPI app — routing, queueing, swaps, admin API, image-gen gate contract. |
| `registry.example.json` | Starting-point schema + 11 sample entries (models themselves not included). |
| `requirements.txt` | `fastapi`, `uvicorn`, `httpx`. |
| `systemd/orchestrator.service` + `.d/override.conf.example` | Service unit and example drop-in. |
| `ops/recover.sh` | Manual recovery if the service won't come up. |
| `README.md`, `LICENSE`, `.gitignore` | Docs and MIT license. |

---

## Features

- OpenAI-compatible: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`
- Two swappable GPU slots (default ports `8011` and `8012`)
- Registry-driven model swaps with `fit_gpu` placement
- Congestion-aware routing (`ORCHESTRATOR_SWAP_CONGESTION_THRESHOLD`)
- Idle swap-back to boot default (`ORCHESTRATOR_IDLE_SWAPBACK_SECONDS`)
- Queueing with timeout (`ORCHESTRATOR_QUEUE_TIMEOUT`)
- Health endpoint reports per-slot state and node mode
- Admin endpoint to stop/start a slot on demand
- Image-gen cooperation gate contract (see above)

**Security posture:** `/admin/*` is **unauthenticated by design** — the trust boundary is your LAN. Bind to a trusted segment, put an authenticating reverse proxy in front, or firewall the port. **Do not expose `:8010` directly to the internet.**

---

## Getting to first inference

1. **Provision the stack** (once per host):
   - Install the NVIDIA driver; verify `nvidia-smi -i 0` and `nvidia-smi -i 1` both work.
   - Build or install `llama-server`; note the absolute path.
   - Download the GGUF files you intend to run; note their absolute paths.
   - *(Only if using Strata)* Install the Strata CLI + engine and prepare a Strata profile.

2. **Install the orchestrator:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Create the data dir and edit the registry:**
   ```bash
   sudo mkdir -p /var/lib/orchestrator
   sudo chown "$USER":"$USER" /var/lib/orchestrator
   cp registry.example.json /var/lib/orchestrator/registry.json
   $EDITOR /var/lib/orchestrator/registry.json
   ```
   In `registry.json`, make every `gguf`/`mmproj` path point at a model you actually have, and set `default_boot` to a pair of model IDs from the `models` block. If you're not using Strata, remove the `qwen3.8-flash-next` entry **and** replace both `default_boot` entries with llama.cpp model IDs.

4. **Run in the foreground first to see errors:**
   ```bash
   ORCHESTRATOR_LLAMA_BIN=/abs/path/to/llama-server python orchestrator.py
   ```
   In another terminal:
   ```bash
   curl -s http://127.0.0.1:8010/health | python3 -m json.tool
   ```
   Both slots should show `state: "ready"` after ~30 s model load. Then send a chat:
   ```bash
   curl http://127.0.0.1:8010/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"<id-from-registry>","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
   ```

5. **(Optional) install as a systemd service:**
   ```bash
   sudo useradd -r -s /usr/sbin/nologin orchestrator   # or reuse an existing service user
   sudo mkdir -p /var/lib/orchestrator
   sudo chown orchestrator:orchestrator /var/lib/orchestrator

   sudo cp systemd/orchestrator.service /etc/systemd/system/
   sudo mkdir -p /etc/systemd/system/orchestrator.service.d
   sudo cp systemd/orchestrator.service.d/override.conf.example \
        /etc/systemd/system/orchestrator.service.d/override.conf
   # edit override.conf: set MemoryMax to ~90% of installed RAM; add ORCHESTRATOR_LLAMA_BIN if not at /usr/local/bin/llama-server

   sudo systemctl daemon-reload
   sudo systemctl enable --now orchestrator.service
   journalctl -u orchestrator.service -f
   ```

---

## Configuration (env vars)

| Environment variable | Default | Description |
|---|---|---|
| `ORCHESTRATOR_DATA_DIR` | `/var/lib/orchestrator` | Main data directory. |
| `ORCHESTRATOR_REGISTRY` | `$ORCHESTRATOR_DATA_DIR/registry.json` | Registry file path. |
| `ORCHESTRATOR_LLAMA_BIN` | `/usr/local/bin/llama-server` | Path to `llama-server`. **Must exist.** |
| `ORCHESTRATOR_LOG_DIR` | `$ORCHESTRATOR_DATA_DIR/logs` | Log directory. |
| `ORCHESTRATOR_MODE_STATE` | `$ORCHESTRATOR_DATA_DIR/node-mode.state` | Node-mode state file (written by an external script — see above). |
| `ORCHESTRATOR_STRATA_BIN` | `/usr/local/bin/strata` | Path to `strata` (only if you use `"backend": "strata"`). |
| `ORCHESTRATOR_STRATA_PROFILE` | `$ORCHESTRATOR_DATA_DIR/profiles/strata-single.json` | Default Strata profile path. |
| `ORCHESTRATOR_WARM_CACHE_SCRIPT` | `$ORCHESTRATOR_DATA_DIR/strata-warm-cache.py` | Optional warm-cache helper. |
| `ORCHESTRATOR_CUDA_RT` | `/opt/llama-cpp-rt` | Prepended to `LD_LIBRARY_PATH` for llama-server children. |
| `ORCHESTRATOR_OWNER` | `orchestrator` | `owned_by` value reported by `/v1/models`. |
| `ORCHESTRATOR_PORT` | `8010` | Listen port. |
| `ORCHESTRATOR_QUEUE_TIMEOUT` | `300` | Seconds to wait for a slot before 503. |
| `ORCHESTRATOR_SWAP_CONGESTION_THRESHOLD` | `2` | In-flight + queued requests for a model on already-loaded slots before an idle slot is swapped to it. Lower = more aggressive dual-GPU loading. |
| `ORCHESTRATOR_IDLE_SWAPBACK_SECONDS` | `300` | Idle seconds before a non-`default_boot` slot is swapped back to its boot default. `0` disables. |
| `ORCHESTRATOR_HEALTH_LENIENT` | unset | `1`/`true`/`yes` treats any ready slot as “ok” for overall `/health` status. |

---

## Routing policy

1. Route to a free slot that already hosts the requested model (round-robin if multiple).
2. If a matching slot is busy, batch onto it until `ORCHESTRATOR_SWAP_CONGESTION_THRESHOLD` requests are in flight/queued for that model; then swap an idle slot to the requested model (respecting `fit_gpu`).
3. If every slot is busy, continuous-batch onto the least-busy matching slot; if no slot hosts the model, queue a swap on the least-busy slot with `ORCHESTRATOR_QUEUE_TIMEOUT`.
4. When a slot becomes idle with a non-`default_boot` model, it is automatically swapped back to its boot default after `ORCHESTRATOR_IDLE_SWAPBACK_SECONDS`.

Slot selection and the `active_requests` reservation run under a single route lock, so two concurrent requests cannot both claim the same free slot.

---

## Registry schema

Top level:

```jsonc
{
  "models": { "<model-id>": { /* entry */ }, ... },
  "default_boot": ["<model-id>", "<model-id>"]   // one entry per slot
}
```

Entry fields (all values are what the orchestrator passes to `llama-server` or to a Strata profile; the exact meaning is llama.cpp's / Strata's):

| Field | Type | Notes |
|---|---|---|
| `gguf` | string | Path to GGUF file. **Must exist on disk** for llama.cpp backends. |
| `mmproj` | string \| null | Vision projector file, if any. |
| `backend` | string | `strata` for Strata; omit for llama.cpp. |
| `strata_profile` | string | Only with `backend: strata`. |
| `ctx_size` | int | Context size in tokens. |
| `gpu_layers` | int | Layers to offload to GPU. |
| `flash_attention` | bool | Passed as `--flash-attn on/off`. |
| `cache_type_k`, `cache_type_v` | string | KV-cache quantisation (`q4_0`, `q8_0`, `f16`, …). |
| `ubatch`, `batch` | int | Batch sizes. |
| `temperature`, `top_p`, `top_k` | num | Sampling defaults. |
| `reasoning` | bool | Whether to parse a reasoning block. |
| `reasoning_format` | string | Format hint (e.g. `deepseek`, `none`). |
| `reasoning_budget` | int | Reasoning token budget. |
| `chat_template_kwargs` | object \| JSON string | Sent to the chat template. |
| `fit_gpu` | int[] | GPU indices this model may load on. |
| `family` | string | Free-form grouping (e.g. `qwen`). |
| `wide_slot` | bool | If true, the model takes **both** GPUs. |
| `attachment` | bool | Advertises image input in `/v1/models`. |
| `tool_call` | bool | Advertises tool-call support. |
| `output_limit` | int | Max output tokens advertised. |
| `use_mtp`, `mtp_n_max`, `mtp_draft_model` | – | MTP speculative decoding. |
| `image_min_tokens` | int | Minimum image tokens (Qwen-VL style). |
| `load_mode` | string | Passed to `--load-mode`. |
| `n_cpu_moe`, `moe_cache_mib`, `cpu_moe`, `n_cpu_ffn` | – | CPU MoE / FFN offload. |
| `split_mode`, `tensor_split`, `override_tensors` | – | Multi-GPU tensor split (advanced). |
| `spec_type`, `spec_draft_n_max` | – | Speculative decoding (advanced). |
| `external_endpoint` | string | Route to an already-running upstream instead of spawning a child. |

Everything not in this list is ignored by the orchestrator.

---

## API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming). |
| `POST` | `/v1/completions` | Completions. |
| `GET` | `/v1/models` | List models from the registry. |
| `GET` | `/health` | Per-slot state, node mode, overall status. |
| `POST` | `/admin/slot/{i}/stop` | Kill the slot's child, drain VRAM, mark `stopped`. |
| `POST` | `/admin/slot/{i}/start` | Load the slot's `default_boot` model. |
| `GET` | `/admin/slot/{i}` | Slot state. |

---

## Operations — recovery

If the service won't come up:

```bash
sudo ops/recover.sh
```

Stops the service, frees port `8010`, restarts, and polls `/health` for up to 30 s. If it still fails:

```bash
sudo journalctl -u orchestrator.service -n 100
cat /var/lib/orchestrator/logs/orchestrator.log
```
