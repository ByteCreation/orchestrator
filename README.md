# orchestrator

One OpenAI-compatible endpoint, two swappable GPU slots, registry-driven model swaps.

This tool provides a FastAPI-compatible proxy that fronts two llama.cpp GPU slots on dual RTX 3090s. It supports model swapping, queueing, and image-gen cooperation.

## Features

- One OpenAI-compatible endpoint (`/v1/chat/completions`, `/v1/completions`)
- Two swappable GPU slots (on ports 8011 and 8012)
- Registry-driven model swaps
- Queueing with `ORCHESTRATOR_QUEUE_TIMEOUT` (default: 300s)
- Image-gen cooperation gate contract: an external writer sets a node-mode state file, orchestrator serves 503 `[IMAGEGEN_ACTIVE]` while non-default

## Quick Start

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create data directory and copy registry:
```bash
sudo mkdir -p /var/lib/orchestrator
sudo chown $USER:$USER /var/lib/orchestrator
cp registry.example.json /var/lib/orchestrator/registry.json
```

3. Run the orchestrator:
```bash
python orchestrator.py
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ORCHESTRATOR_DATA_DIR` | `/var/lib/orchestrator` | Main data directory |
| `ORCHESTRATOR_REGISTRY` | `/var/lib/orchestrator/registry.json` | Registry file path |
| `ORCHESTRATOR_LLAMA_BIN` | `/usr/local/bin/llama-server` | Path to llama-server binary |
| `ORCHESTRATOR_LOG_DIR` | `/var/lib/orchestrator/logs` | Log directory |
| `ORCHESTRATOR_MODE_STATE` | `/var/lib/orchestrator/node-mode.state` | Node mode state file |
| `ORCHESTRATOR_STRATA_BIN` | `/usr/local/bin/strata` | Path to strata binary |
| `ORCHESTRATOR_STRATA_PROFILE` | `/var/lib/orchestrator/profiles/strata-single.json` | Default strata profile |
| `ORCHESTRATOR_WARM_CACHE_SCRIPT` | `/var/lib/orchestrator/strata-warm-cache.py` | Path to warm cache script |
| `ORCHESTRATOR_CUDA_RT` | `/opt/llama-cpp-rt` | CUDA runtime directory |
| `ORCHESTRATOR_OWNER` | `orchestrator` | Owner name for models |
| `ORCHESTRATOR_PORT` | `8010` | Listen port for the OpenAI-compatible endpoint |
| `ORCHESTRATOR_QUEUE_TIMEOUT` | `300` | Queue timeout in seconds |
| `ORCHESTRATOR_HEALTH_LENIENT` | `""` | Set to `1`, `true`, or `yes` to treat any ready slot as "ok" |

## Registry Schema

| Field | Type | Description |
|------|--------|----------------|
| `gguf` | string | Path to GGUF model file |
| `mmproj` | string | Path to multimodal projection file |
| `ctx_size` | int | Context size |
| `gpu_layers` | int | Number of GPU layers |
| `flash_attention` | bool | Use flash attention |
| `cache_type_k` | string | Cache type for K |
| `cache_type_v` | string | Cache type for V |
| `ubatch` | int | U-batch size |
| `batch` | int | Batch size |
| `temperature` | float | Temperature |
| `top_p` | float | Top-p |
| `top_k` | int | Top-k |
| `reasoning` | bool | Enable reasoning |
| `reasoning_format` | string | Reasoning format |
| `fit_gpu` | array | GPU IDs this model fits on |
| `family` | string | Model family |
| `wide_slot` | bool | Use both GPUs |
| `attachment` | bool | Supports attachments |
| `tool_call` | bool | Supports tool calls |
| `output_limit` | int | Output limit |
| `backend` | string | Backend type (`strata`) |
| `strata_profile` | string | Strata profile path |
| `use_mtp` | bool | Use MTP speculative decoding |
| `mtp_n_max` | int | MTP max N |
| `image_min_tokens` | int | Min tokens for image models |
| `reasoning_budget` | int | Reasoning budget |
| `load_mode` | string | Load mode |
| `n_cpu_moe` | int | CPU MoE |
| `moe_cache_mib` | int | MoE cache size |
| `cpu_moe` | bool | CPU MoE |
| `n_cpu_ffn` | int | CPU FFN layers |
| `split_mode` | string | Split mode |
| `tensor_split` | array | Tensor split |
| `override_tensors` | string | Override tensors |
| `spec_type` | string | Speculative decoding type |
| `spec_draft_n_max` | int | Speculative draft max N |
| `mtp_draft_model` | string | MTP draft model |
| `chat_template_kwargs` | object | Chat template kwargs |
| `external_endpoint` | string | External endpoint |

## API Surface

- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Completions
- `GET /v1/models` - List models
- `GET /health` - Health check
- `POST /admin/slot/{slot_index}/stop` - Stop slot
- `POST /admin/slot/{slot_index}/start` - Start slot
- `GET /admin/slot/{slot_index}` - Get slot status

**Security posture:** `/admin/*` is unauthenticated by design — the trust
boundary is your LAN. Bind the service to a trusted network segment, put a
reverse proxy with auth in front of it, or keep the port firewalled. Do not
expose `:8010` directly to the internet.

## Systemd Installation

1. Copy the service file:
```bash
sudo cp systemd/orchestrator.service /etc/systemd/system/
```

2. Create the data directory:
```bash
sudo mkdir -p /var/lib/orchestrator
sudo chown orchestrator:orchestrator /var/lib/orchestrator
```

3. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable orchestrator.service
sudo systemctl start orchestrator.service
```

## Operations

### Recovery

Use the recovery script to manually recover if the orchestrator won't start:

```bash
sudo ops/recover.sh
```

This script will:
1. Stop the orchestrator service
2. Kill any stuck processes on port 8010
3. Start the orchestrator service
4. Verify that it's running