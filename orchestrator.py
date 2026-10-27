#!/usr/bin/env python3
"""orchestrator — OpenAI-compatible proxy over two llama-server GPU slots.

Design:
  * One model per GPU at a time; two slots on :8011 (GPU0) and :8012 (GPU1).
  * On boot, load registry.default_boot (dual Swift by default).
  * Requests are routed to a slot that already hosts the requested model; if none,
    the least-busy slot is picked and, if busy, the request is queued until idle.
  * Swaps only happen when active_requests == 0 for a slot.
  * Server-side reasoning/thinking is controlled by the registry.
  * No Ollama fallback in Phase 1 — unknown models return 404.

Config paths:
  /var/lib/orchestrator/registry.json   model registry
  /var/lib/orchestrator/logs/     logs + optional state.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("ORCHESTRATOR_DATA_DIR", "/var/lib/orchestrator"))
REGISTRY_PATH = Path(os.environ.get("ORCHESTRATOR_REGISTRY", str(DATA_DIR / "registry.json")))
LLAMA_BIN = Path(os.environ.get("ORCHESTRATOR_LLAMA_BIN", "/usr/local/bin/llama-server"))
LOG_DIR = Path(os.environ.get("ORCHESTRATOR_LOG_DIR", str(DATA_DIR / "logs")))
STATE_FILE = LOG_DIR / "state.json"
MODE_STATE_FILE = Path(os.environ.get("ORCHESTRATOR_MODE_STATE", str(DATA_DIR / "node-mode.state")))
STRATA_BIN = Path(os.environ.get("ORCHESTRATOR_STRATA_BIN", "/usr/local/bin/strata"))
DEFAULT_STRATA_PROFILE = Path(os.environ.get("ORCHESTRATOR_STRATA_PROFILE", str(DATA_DIR / "profiles/strata-single.json")))

SLOT_PORTS = [8011, 8012]
SLOT_GPUS = [0, 1]
ORCHESTRATOR_PORT = int(os.environ.get("ORCHESTRATOR_PORT", "8010"))
DEFAULT_QUEUE_TIMEOUT = 300
VRAM_DRAIN_TIMEOUT = 60
VRAM_DRAIN_THRESHOLD_MIB = 1024
HEALTH_POLL_INTERVAL = 1
HEALTH_START_TIMEOUT = 300
VRAM_POLL_INTERVAL = 2
GRACEFUL_SHUTDOWN_SECONDS = 30

# Image-gen cooperation: node-mode.state records the current node
# mode here. While an image/music-gen job owns a GPU, do not queue model
# swaps against it; fail fast with an explanatory 503 (token
# IMAGEGEN_ACTIVE) so agents wait for the automatic restore instead of
# hanging until the client times out.
MODE_STATE_CACHE_TTL = 2.0
BUSY_MODES = ("imagegen", "imagegen-single", "imagegen-single-fallback", "musicgen")

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "orchestrator.log", maxBytes=10 * 1024 * 1024, backupCount=5
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("orchestrator")

# -----------------------------------------------------------------------------
# Node mode (written by node-mode / imggen-run)
# -----------------------------------------------------------------------------
_node_mode_cache: tuple = (0.0, "unknown")


def get_node_mode() -> str:
    """Return the current node mode (default, imagegen, ...)."""
    global _node_mode_cache
    now = time.time()
    if now - _node_mode_cache[0] >= MODE_STATE_CACHE_TTL:
        try:
            mode = MODE_STATE_FILE.read_text().strip() or "unknown"
        except OSError:
            mode = "unknown"
        _node_mode_cache = (now, mode)
    return _node_mode_cache[1]


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
class Registry:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {}
        self.refresh()

    def refresh(self) -> None:
        with self.path.open() as f:
            self.data = json.load(f)
        logger.info("Loaded registry with %d models", len(self.data.get("models", {})))

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get("models", {}).get(model_id)

    @property
    def models(self) -> Dict[str, Any]:
        return self.data.get("models", {})

    def fit_gpus(self, model_id: str) -> Optional[List[int]]:
        meta = self.get(model_id)
        if not meta:
            return None
        return meta.get("fit_gpu")

    @property
    def default_boot(self) -> List[str]:
        return self.data.get("default_boot", [])


# -----------------------------------------------------------------------------
# Slot state
# -----------------------------------------------------------------------------
@dataclass
class Slot:
    index: int
    gpu: int
    port: int
    loaded_model: Optional[str] = None
    state: str = "init"  # init, loading, ready, error, stopping, stopped
    active_requests: int = 0
    process: Optional[subprocess.Popen] = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)

    def url(self, path: str = "") -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def is_operational(self) -> bool:
        return self.state in ("init", "ready", "loading")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_vram_mib(gpu: int) -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        logger.exception("nvidia-smi query failed for gpu %d", gpu)
        return 0


async def wait_for_vram_drain(gpu: int, timeout: int = VRAM_DRAIN_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        used = get_vram_mib(gpu)
        if used < VRAM_DRAIN_THRESHOLD_MIB:
            logger.info("GPU %d VRAM drained to %d MiB", gpu, used)
            return True
        await asyncio.sleep(VRAM_POLL_INTERVAL)
    logger.warning("GPU %d VRAM did not drain below %d MiB within %ds (last=%d)",
                   gpu, VRAM_DRAIN_THRESHOLD_MIB, timeout, get_vram_mib(gpu))
    return False


async def wait_for_all_vram_drain(gpus: List[int], timeout: int = VRAM_DRAIN_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(get_vram_mib(g) < VRAM_DRAIN_THRESHOLD_MIB for g in gpus):
            logger.info("All requested GPUs drained: %s", gpus)
            return True
        await asyncio.sleep(VRAM_POLL_INTERVAL)
    logger.warning("Not all GPUs drained within %ds", timeout)
    return False


async def wait_for_health(slot: Slot, timeout: int = HEALTH_START_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < deadline:
            # If the child process died, abort immediately.
            if slot.process and slot.process.poll() is not None:
                logger.error("Slot %d llama-server exited (code %s) before health OK",
                             slot.index, slot.process.returncode)
                return False
            try:
                r = await client.get(slot.url("/health"))
                if r.status_code == 200:
                    logger.info("Slot %d (GPU %d port %d) health OK", slot.index, slot.gpu, slot.port)
                    return True
            except Exception:
                pass
            await asyncio.sleep(HEALTH_POLL_INTERVAL)
    logger.warning("Slot %d health check timed out after %ds", slot.index, timeout)
    return False


def build_llama_args(model_id: str, meta: Dict[str, Any], slot: Slot) -> List[str]:
    args = [
        str(LLAMA_BIN),
        "--model", str(meta["gguf"]),
        "--ctx-size", str(meta["ctx_size"]),
        "--gpu-layers", str(meta["gpu_layers"]),
        "--flash-attn", "on" if meta.get("flash_attention") else "off",
        "--cache-type-k", str(meta["cache_type_k"]),
        "--cache-type-v", str(meta["cache_type_v"]),
        "--ubatch-size", str(meta["ubatch"]),
        "--batch-size", str(meta["batch"]),
        "--temp", str(meta["temperature"]),
        "--top-p", str(meta["top_p"]),
        "--top-k", str(meta["top_k"]),
        "--jinja",
        "--host", "127.0.0.1",
        "--port", str(slot.port),
        "--alias", model_id,
    ]

    mmproj = meta.get("mmproj")
    if mmproj:
        args += [
            "--mmproj", str(mmproj),
            "--mmproj-device", "none",
        ]

    chat_kwargs = meta.get("chat_template_kwargs")
    if chat_kwargs:
        if isinstance(chat_kwargs, dict):
            chat_kwargs = json.dumps(chat_kwargs, separators=(',', ':'))
        args += ["--chat-template-kwargs", str(chat_kwargs)]

    reasoning = meta.get("reasoning", False)
    reasoning_format = meta.get("reasoning_format", "none")
    args += [
        "--reasoning", "on" if reasoning else "off",
        "--reasoning-format", str(reasoning_format),
    ]

    if "reasoning_budget" in meta:
        args += ["--reasoning-budget", str(meta["reasoning_budget"])]

    # b11480 has --moe-cache-mib (initial single-GPU implementation).
    if meta.get("moe_cache_mib"):
        args += ["--moe-cache-mib", str(meta["moe_cache_mib"])]

    # v0.7.0+ MoE offload: --cpu-moe (all experts on CPU) or --n-cpu-moe N (partial).
    # cpu_moe takes precedence; if absent, fall back to n_cpu_moe.
    if meta.get("cpu_moe"):
        args += ["--cpu-moe"]
    else:
        n_cpu_moe = meta.get("n_cpu_moe")
        if n_cpu_moe is None and meta.get("family") == "qwen":
            n_cpu_moe = 0
        if n_cpu_moe is not None:
            args += ["--n-cpu-moe", str(n_cpu_moe)]

    # --load-mode: v0.7.0+ replacement for --no-mmap.
    if meta.get("load_mode"):
        args += ["--load-mode", str(meta["load_mode"])]

    if meta.get("n_cpu_ffn") is not None:
        args += ["--n-cpu-ffn", str(meta["n_cpu_ffn"])]

    # Wide-slot / multi-GPU controls (only emitted when the model asks for them)
    split_mode = meta.get("split_mode")
    if split_mode:
        args += ["--split-mode", split_mode]
        if split_mode == "tensor":
            # Tensor split in b11480 does not allow quantized KV for all archs.
            args += ["--flash-attn", "on"]
            if meta.get("cache_type_k") in ("q4_0", "q8_0", "q4_1", "iq4_nl", "q5_0", "q5_1"):
                args[args.index("--cache-type-k") + 1] = "f16"
            if meta.get("cache_type_v") in ("q4_0", "q8_0", "q4_1", "iq4_nl", "q5_0", "q5_1"):
                args[args.index("--cache-type-v") + 1] = "f16"
        if "tensor_split" in meta:
            args += ["--tensor-split", str(meta["tensor_split"])]

    override_tensors = meta.get("override_tensors")
    if override_tensors:
        args += ["--override-tensor", str(override_tensors)]

    # MTP speculative decoding for qwen35/qwen4exp.
    # Speculative decoding: MTP takes precedence, otherwise honor spec_type.
    if meta.get("use_mtp"):
        args += [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(meta.get("mtp_n_max", 2)),
        ]
        if meta.get("mtp_draft_model"):
            args += [
                "--spec-draft-model", str(meta["mtp_draft_model"]),
                "--n-gpu-layers-draft", "99",
            ]
    elif meta.get("spec_type"):
        args += [
            "--spec-type", str(meta["spec_type"]),
            "--spec-draft-n-max", str(meta.get("spec_draft_n_max", 3)),
        ]

    if meta.get("image_min_tokens"):
        args += ["--image-min-tokens", str(meta["image_min_tokens"])]

    return args


def write_strata_profile(model_id: str, meta: Dict[str, Any], slot: Slot) -> Path:
    """Generate a temporary Strata run-config that targets the chosen GPU/port."""
    base_path = Path(meta.get("strata_profile", DEFAULT_STRATA_PROFILE))
    profile = json.loads(base_path.read_text())
    profile["host"] = "127.0.0.1"
    profile["port"] = slot.port
    if meta.get("wide_slot"):
        profile["gpu"] = list(SLOT_GPUS)
    else:
        profile["gpu"] = slot.gpu
    profile["model_name"] = model_id
    profile["log"] = str(LOG_DIR / f"strata-slot{slot.index}-{model_id}.engine.log")
    profile["open_browser"] = False
    out_path = Path(f"/tmp/strata-slot{slot.index}-{model_id}.json")
    out_path.write_text(json.dumps(profile, indent=2))
    return out_path


def build_strata_args(model_id: str, meta: Dict[str, Any], slot: Slot) -> List[str]:
    profile_path = write_strata_profile(model_id, meta, slot)
    parallel = 2 if meta.get("wide_slot") else 1
    return [
        str(STRATA_BIN),
        "serve",
        "--profile",
        str(profile_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(slot.port),
        "--parallel",
        str(parallel),
    ]


async def warm_strata_cache(max_gb: Optional[float] = None) -> None:
    """Pre-warm Strata model files in the Linux page cache before loading.

    This overlaps with VRAM drain so it does not add wall-clock time unless
    the cache warmer is slower than the drain itself.
    """
    if max_gb is None:
        max_gb = float(os.environ.get("STRATA_WARM_CACHE_MAX_GB", "34"))
    script = Path(os.environ.get("ORCHESTRATOR_WARM_CACHE_SCRIPT", str(DATA_DIR / "strata-warm-cache.py")))
    if not script.exists():
        logger.warning("Cache warmer script not found: %s", script)
        return
    try:
        logger.info("Warming Strata page cache (max %.1f GiB)...", max_gb)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script), "--max-gb", str(max_gb),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
        for line in stdout.decode(errors="replace").splitlines():
            logger.info("[warm-cache] %s", line)
    except asyncio.TimeoutError:
        logger.warning("Strata cache warmer timed out after 300 s; proceeding anyway")
    except Exception:
        logger.exception("Strata cache warmer failed; proceeding anyway")


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

class Orchestrator:
    def __init__(self):
        self.registry = Registry(REGISTRY_PATH)
        self.slots: List[Slot] = [Slot(i, SLOT_GPUS[i], SLOT_PORTS[i]) for i in range(len(SLOT_PORTS))]
        self.queue_timeout = float(os.environ.get("ORCHESTRATOR_QUEUE_TIMEOUT", DEFAULT_QUEUE_TIMEOUT))
        self.shutting_down = False
        self.http_client = httpx.AsyncClient(timeout=None)
        self._tasks: set = set()
        self._route_counter = 0
        self._route_lock = asyncio.Lock()

    # --- llama-server process management ---

    async def _kill_slot(self, slot: Slot) -> None:
        """Kill the child process on this slot. Caller must hold slot.lock."""
        if slot.process and slot.process.poll() is None:
            logger.info("Terminating llama-server on slot %d (pid %s)", slot.index, slot.process.pid)
            try:
                slot.process.terminate()
                await asyncio.wait_for(asyncio.to_thread(slot.process.wait), timeout=10)
            except asyncio.TimeoutError:
                logger.warning("Slot %d did not terminate in 10s; killing", slot.index)
                slot.process.kill()
                await asyncio.to_thread(slot.process.wait)
        slot.process = None
        slot.state = "init"
        slot.loaded_model = None

    async def _start_model(self, slot: Slot, model_id: str) -> bool:
        meta = self.registry.get(model_id)
        if not meta:
            logger.error("Model %s not in registry", model_id)
            slot.state = "error"
            return False

        slot.ready_event.clear()
        try:
            # Wide-slot models consume both GPUs: hold both slot locks in index
            # order to avoid deadlocks with concurrent boot/queue workers.
            if meta.get("wide_slot"):
                first, second = sorted([slot, self.slots[1 - slot.index]], key=lambda s: s.index)
                async with first.lock:
                    async with second.lock:
                        ok = await self._start_model_locked(slot, model_id, meta)
            else:
                async with slot.lock:
                    ok = await self._start_model_locked(slot, model_id, meta)
        finally:
            slot.ready_event.set()
        return ok

    async def _start_strata_locked(self, slot: Slot, model_id: str, meta: Dict[str, Any]) -> bool:
        """Start Strata on a single GPU slot (or both GPUs for wide_slot)."""
        if slot.state == "ready" and slot.loaded_model == model_id:
            return True
        await self._kill_slot(slot)

        # Warm the page cache in parallel with VRAM drain so the model files
        # are resident in RAM before Strata starts reading them.
        warm_task: Optional[asyncio.Task] = asyncio.create_task(warm_strata_cache())

        if meta.get("wide_slot"):
            sibling = self.slots[1 - slot.index]
            await self._kill_slot(sibling)
            await wait_for_all_vram_drain(SLOT_GPUS, timeout=VRAM_DRAIN_TIMEOUT)
        else:
            await wait_for_vram_drain(slot.gpu, timeout=VRAM_DRAIN_TIMEOUT)

        # Ensure the warmer finishes before we start Strata; if it is slower
        # than the drain we pay its remaining time here.
        await warm_task

        args = build_strata_args(model_id, meta, slot)
        log_path = LOG_DIR / f"strata-slot{slot.index}-{model_id}.log"
        logger.info("Starting Strata on slot %d (GPU %d, port %d) for %s: %s",
                    slot.index, slot.gpu, slot.port, model_id, " ".join(args))
        slot.state = "loading"
        slot.loaded_model = model_id
        try:
            env = os.environ.copy()
            env["STRATA_DATA_DIR"] = os.environ.get("STRATA_DATA_DIR", str(DATA_DIR / "strata"))
            env["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            # Strata expert-arena backing file (engine knob STRATA_ARENA_MMAP): the arena
            # becomes file-backed, "not locked, not pinned" — reclaimable under pressure,
            # which is what makes dual Strata possible on this host. First start writes
            # the file, the next start maps it. One file per slot so instances never share
            # a backing image. Gated: only active when STRATA_ARENA_MMAP_BASE is set in
            # the service environment (see orchestrator drop-in strata-arena-mmap.conf).
            arena_base = env.get("STRATA_ARENA_MMAP_BASE")
            if arena_base:
                env["STRATA_ARENA_MMAP"] = f"{arena_base}/slot{slot.index}.arena"
            f = log_path.open("wb")
            slot.process = subprocess.Popen(
                args,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except Exception:
            logger.exception("Failed to start Strata on slot %d", slot.index)
            slot.state = "error"
            return False

        if await wait_for_health(slot, timeout=HEALTH_START_TIMEOUT):
            slot.loaded_model = model_id
            slot.state = "ready"
            logger.info("Strata ready on slot %d (GPU %d, port %d) for %s",
                        slot.index, slot.gpu, slot.port, model_id)
            return True
        else:
            logger.error("Strata on slot %d failed to become healthy for %s", slot.index, model_id)
            await self._kill_slot(slot)
            slot.state = "error"
            return False

    async def _start_model_locked(self, slot: Slot, model_id: str, meta: Dict[str, Any]) -> bool:
        if slot.state == "ready" and slot.loaded_model == model_id:
            return True

        # Strata-backed models are managed as first-class slots.
        if meta.get("backend") == "strata":
            return await self._start_strata_locked(slot, model_id, meta)

        await self._kill_slot(slot)

        # Wide-slot models consume both GPUs: kill the sibling slot too.
        if meta.get("wide_slot"):
            sibling = self.slots[1 - slot.index]
            await self._kill_slot(sibling)

        # Wait for VRAM to drain after killing previous instance.
        if meta.get("wide_slot"):
            await wait_for_all_vram_drain(SLOT_GPUS, timeout=VRAM_DRAIN_TIMEOUT)
        else:
            await wait_for_vram_drain(slot.gpu, timeout=VRAM_DRAIN_TIMEOUT)

        env = os.environ.copy()
        # Wide-slot models run across all GPUs; normal models stay isolated to one GPU.
        if meta.get("wide_slot"):
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in SLOT_GPUS)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        # Preserve the llama.cpp runtime library path used by the standalone script.
        rt = Path(os.environ.get("ORCHESTRATOR_CUDA_RT", "/opt/llama-cpp-rt"))
        if rt.exists():
            env["LD_LIBRARY_PATH"] = (
                f"{rt}/cuda132x/usr/local/cuda-13.2/targets/x86_64-linux/lib:"
                f"{rt}/cuda132x/usr/local/cuda-13.2/nvvm/lib:"
                f"{rt}/gcc14/usr/lib"
            )

        args = build_llama_args(model_id, meta, slot)
        log_path = LOG_DIR / f"slot{slot.index}-{model_id}.log"
        logger.info("Starting slot %d with model %s on GPU %d: %s", slot.index, model_id, slot.gpu, " ".join(args))
        slot.state = "loading"
        slot.loaded_model = model_id
        try:
            f = log_path.open("wb")
            slot.process = subprocess.Popen(
                args,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        except Exception:
            logger.exception("Failed to start llama-server for slot %d", slot.index)
            slot.state = "error"
            return False

        if await wait_for_health(slot, timeout=HEALTH_START_TIMEOUT):
            slot.loaded_model = model_id
            slot.state = "ready"
            return True
        else:
            logger.error("Slot %d failed to become healthy with model %s", slot.index, model_id)
            await self._kill_slot(slot)
            slot.state = "error"
            return False

    # --- Boot ---

    async def boot(self) -> None:
        defaults = self.registry.default_boot
        if len(defaults) < len(self.slots):
            logger.warning("default_boot shorter than slot count; padding with first entry")
            defaults = defaults + [defaults[0]] * (len(self.slots) - len(defaults))
        logger.info("Booting slots with defaults: %s", defaults[:len(self.slots)])
        results = await asyncio.gather(
            *(self._start_model(slot, defaults[i]) for i, slot in enumerate(self.slots)),
            return_exceptions=True,
        )
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.exception("Slot %d boot failed", i)
                self.slots[i].state = "error"
        logger.info("Boot complete")

    async def boot_background(self) -> None:
        try:
            await self.boot()
        except Exception:
            logger.exception("Background boot failed")

    # --- Routing ---

    def _model_fits_slot(self, model_id: str, slot: Slot) -> bool:
        """Return True if the model's fit_gpu list permits this slot's GPU."""
        fit = self.registry.fit_gpus(model_id)
        if not fit:
            return True
        return slot.gpu in fit

    def _round_robin_pick(self, slots: List[Slot]) -> Slot:
        """Pick among equally-suitable slots with a global round-robin counter."""
        slots = sorted(slots, key=lambda s: s.index)
        chosen = slots[self._route_counter % len(slots)]
        self._route_counter += 1
        return chosen

    async def _ensure_model_loaded(self, model_id: str) -> Slot:
        """Choose and reserve a slot atomically, then load the model if needed.

        Routing goals:
          1. Prefer a free slot that already hosts the requested model.
          2. Otherwise swap a free idle slot to the model so both GPUs stay busy.
          3. If every slot is busy, join the least-busy slot already hosting the
             model (llama-server continuous batching), or queue a swap.
        """
        chosen: Optional[Slot] = None
        swap = False
        queue_event: Optional[asyncio.Event] = None
        wait_event: Optional[asyncio.Event] = None

        # Hold the route lock while we snapshot load and reserve a slot. This
        # prevents concurrent requests from piling onto the same slot before
        # active_requests is updated.
        async with self._route_lock:
            # Image-gen cooperation: while node-mode reports an image/music
            # gen job, only a model already loaded and ready (e.g. the GPU 0 LLM
            # that survives imagegen-single) may be served. A request that would
            # swap a GPU owned by the job gets a fast, self-explanatory 503
            # instead of hanging until the client times out.
            node_mode = get_node_mode()
            if node_mode in BUSY_MODES and not any(
                s.state == "ready" and s.loaded_model == model_id for s in self.slots
            ):
                slots_desc = ", ".join(
                    f"slot {s.index}/GPU {s.gpu}: {s.state}"
                    + (f" ({s.loaded_model})" if s.loaded_model else "")
                    for s in self.slots
                )
                logger.info(
                    "Rejecting %s: node mode '%s' active (IMAGEGEN_ACTIVE)",
                    model_id, node_mode,
                )
                raise HTTPException(
                    status_code=503,
                    headers={"Retry-After": "60"},
                    detail=(
                        f"[IMAGEGEN_ACTIVE] Nothing has crashed: node is "
                        f"intentionally running '{node_mode}' — a local "
                        f"image/music-generation job currently owns the GPU(s), so "
                        f"'{model_id}' cannot be loaded right now. The job restores "
                        f"full LLM service automatically when it finishes (usually "
                        f"1-3 minutes). Please wait and retry this exact request: "
                        f"poll `curl -s http://<host>:8010/health` every "
                        f"~30s until every slot reports state=ready (or `cat "
                        f"{MODE_STATE_FILE}` shows 'default'). Current state: "
                        f"{slots_desc}."
                    ),
                )

            # If any slot is running a wide-slot model, no other model can load.
            for s in self.slots:
                if s.state == "ready" and s.loaded_model:
                    meta = self.registry.get(s.loaded_model)
                    if meta and meta.get("wide_slot"):
                        if s.loaded_model == model_id:
                            s.active_requests += 1
                            logger.info("Routing %s to wide slot %d (already loaded)", model_id, s.index)
                            return s
                        raise HTTPException(
                            status_code=503,
                            detail=f"Node is locked by wide-slot model '{s.loaded_model}'; only that model can run",
                        )

            candidates = [
                s for s in self.slots
                if self._model_fits_slot(model_id, s) and s.is_operational()
            ]
            if not candidates:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model '{model_id}' cannot run on any available GPU",
                )

            # 1. Free slot already loaded with the requested model.
            free_matching = [
                s for s in candidates
                if s.state == "ready" and s.loaded_model == model_id and s.active_requests == 0
            ]
            if free_matching:
                chosen = self._round_robin_pick(free_matching)
                chosen.active_requests += 1
                logger.info(
                    "Routing %s to free already-loaded slot %d (GPU %d)",
                    model_id, chosen.index, chosen.gpu,
                )
                return chosen

            # 1b. A slot is already loading the requested model. Wait for it
            #     rather than starting a second copy on another GPU.
            loading_matching = [
                s for s in candidates
                if s.state == "loading" and s.loaded_model == model_id
            ]
            if loading_matching:
                chosen = min(loading_matching, key=lambda s: (s.active_requests, s.index))
                chosen.active_requests += 1
                wait_event = chosen.ready_event
                logger.info(
                    "Routing %s to slot %d (GPU %d) which is already loading the model; waiting",
                    model_id, chosen.index, chosen.gpu,
                )
                # fall through to await the slot outside the lock.

            # 2. Free idle slot with a different model (or empty) -> swap to target.
            #    Congestion gate: if a ready slot already hosts this model but is
            #    only momentarily busy (empty queue, <2 active), do NOT swap --
            #    fall through to step 3 and continuous-batch onto it. A swap costs
            #    a full model reload (~1 min) and loses to queueing unless the
            #    matching slot is genuinely congested (e.g. simultaneous burst).
            elif chosen is None:
                ready_matching = [
                    s for s in candidates
                    if s.state == "ready" and s.loaded_model == model_id
                ]
                allow_swap = (
                    not ready_matching
                    or any(
                        s.queue.qsize() >= 1 or s.active_requests >= 2
                        for s in ready_matching
                    )
                )
                free_other = [
                    s for s in candidates
                    if s.state in ("ready", "init")
                    and s.loaded_model != model_id
                    and s.active_requests == 0
                ] if allow_swap else []
                if free_other:
                    chosen = self._round_robin_pick(free_other)
                    # Reserve the slot while the swap runs so another request does
                    # not grab it.
                    chosen.active_requests += 1
                    chosen.state = "loading"
                    chosen.loaded_model = None
                    swap = True
                    logger.info(
                        "Swapping idle slot %d (GPU %d) to %s",
                        chosen.index, chosen.gpu, model_id,
                    )
                    # fall through to perform the swap outside the lock.

            if chosen is None:
                # 3. All slots are busy. Continuous-batch on the least-busy slot
                #    that already hosts the model, if any.
                busy_matching = [
                    s for s in candidates
                    if s.state == "ready" and s.loaded_model == model_id
                ]
                if busy_matching:
                    chosen = min(busy_matching, key=lambda s: (s.active_requests, s.index))
                    chosen.active_requests += 1
                    logger.info(
                        "Routing %s to busy already-loaded slot %d (GPU %d, active=%d)",
                        model_id, chosen.index, chosen.gpu, chosen.active_requests,
                    )
                    return chosen

                # No matching model loaded anywhere and all slots busy: queue a swap.
                chosen = min(candidates, key=lambda s: (s.active_requests, s.index))
                queue_event = asyncio.Event()
                chosen.queue.put_nowait((model_id, queue_event))
                logger.info(
                    "All slots busy; queued swap of %s on slot %d (active=%d)",
                    model_id, chosen.index, chosen.active_requests,
                )
                # fall through to wait for the worker outside the lock.

        # Wait for a slot that is already loading the requested model. This
        # keeps the other GPU's model intact and avoids duplicate loads.
        if wait_event:
            try:
                await asyncio.wait_for(wait_event.wait(), timeout=self.queue_timeout)
            except asyncio.TimeoutError:
                chosen.active_requests = max(0, chosen.active_requests - 1)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Slot {chosen.index} took longer than {int(self.queue_timeout)}s "
                        f"to finish loading '{model_id}'"
                    ),
                )
            if chosen.loaded_model != model_id or chosen.state != "ready":
                chosen.active_requests = max(0, chosen.active_requests - 1)
                raise HTTPException(
                    status_code=503,
                    detail=f"Slot {chosen.index} failed to load '{model_id}'",
                )
            return chosen

        # Swap on the reserved idle slot outside the route lock so other
        # requests can be routed while llama-server loads.
        if swap:
            ok = await self._start_model(chosen, model_id)
            if not ok:
                chosen.active_requests = max(0, chosen.active_requests - 1)
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load '{model_id}' on slot {chosen.index}",
                )
            return chosen

        if queue_event:
            try:
                await asyncio.wait_for(queue_event.wait(), timeout=self.queue_timeout)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"No free GPU slot available for model '{model_id}' "
                        f"within {int(self.queue_timeout)}s; all slots are currently busy"
                    ),
                )
            if chosen.loaded_model != model_id or chosen.state != "ready":
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"No free GPU slot available for model '{model_id}'; "
                        f"slot {chosen.index} could not be loaded"
                    ),
                )
            return chosen

        # Should never reach here.
        raise HTTPException(status_code=503, detail="Unable to route request")

    # --- Queue worker per slot ---

    async def _slot_worker(self, slot: Slot) -> None:
        logger.info("Starting queue worker for slot %d", slot.index)
        while not self.shutting_down:
            try:
                model_id, event = await asyncio.wait_for(slot.queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            # Wait until slot is idle before swapping.
            while slot.active_requests > 0 and not self.shutting_down:
                await asyncio.sleep(0.5)
            if self.shutting_down:
                event.set()
                continue
            # Reserve the slot for the request that queued the swap, then load.
            slot.active_requests += 1
            slot.state = "loading"
            slot.loaded_model = None
            ok = await self._start_model(slot, model_id)
            if not ok:
                logger.error("Queued swap of %s on slot %d failed", model_id, slot.index)
                slot.active_requests = max(0, slot.active_requests - 1)
                # _start_model leaves the slot in 'init' after killing the failed process.
            event.set()

    async def start_workers(self) -> None:
        for slot in self.slots:
            t = asyncio.create_task(self._slot_worker(slot))
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

    # --- Request forwarding ---

    async def _forward(self, slot: Slot, request: Request) -> Response:
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host",)}
        method = request.method
        path = request.url.path
        if request.url.query:
            path += "?" + request.url.query
        target = slot.url(path)

        is_stream = False
        try:
            data = json.loads(body)
            is_stream = data.get("stream", False)
        except Exception:
            pass

        start = time.time()
        try:
            return await self._do_forward(slot, request, body, headers, method, path, target, is_stream, start)
        except httpx.ConnectError as exc:
            # The chosen slot's llama-server is temporarily unreachable (mid-swap,
            # child crashed, port not yet listening). Decrement the active request
            # counter so the slot isn't considered permanently busy, then return
            # 503 so the client retries on a healthy slot.
            slot.active_requests = max(0, slot.active_requests - 1)
            elapsed = time.time() - start
            logger.warning(
                "Upstream connect error on slot %d (%s) after %.3fs (active=%d): %s",
                slot.index, target, elapsed, slot.active_requests, exc,
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Upstream slot {slot.index} for model '{slot.loaded_model}' "
                    f"is temporarily unreachable ({exc}); retry in a moment"
                ),
            ) from exc

    async def _do_forward(
        self,
        slot: Slot,
        request: Request,
        body: bytes,
        headers: Dict[str, str],
        method: str,
        path: str,
        target: str,
        is_stream: bool,
        start: float,
    ) -> Response:
        if is_stream:
            # active_requests was already reserved in _ensure_model_loaded.
            # Decrement only when the stream generator is fully consumed/closed.
            # Use build_request + send(stream=True) so we can inspect the upstream
            # status before committing to a StreamingResponse; on error we return a
            # plain Response with the real status code and body.
            upstream_req = self.http_client.build_request(
                method=method, url=target, headers=headers, content=body
            )
            resp = await self.http_client.send(upstream_req, stream=True)
            if resp.status_code >= 400:
                error_body = await resp.aread()
                await resp.aclose()
                slot.active_requests -= 1
                elapsed = time.time() - start
                logger.info(
                    "Streaming request to slot %d failed in %.3fs (status=%d, active=%d)",
                    slot.index, elapsed, resp.status_code, slot.active_requests,
                )
                return Response(
                    content=error_body,
                    status_code=resp.status_code,
                    headers={
                        "content-type": resp.headers.get("content-type", "application/json")
                    },
                )

            async def stream_response() -> AsyncIterator[bytes]:
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                except Exception as e:
                    logger.warning("Streaming request to slot %d aborted: %s", slot.index, e)
                finally:
                    await resp.aclose()
                    slot.active_requests -= 1
                    elapsed = time.time() - start
                    logger.info("Streaming request to slot %d completed in %.3fs (active=%d)", slot.index, elapsed, slot.active_requests)
            return StreamingResponse(stream_response(), media_type="text/plain")
        else:
            try:
                resp = await self.http_client.request(
                    method=method, url=target, headers=headers, content=body
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers={k: v for k, v in resp.headers.items() if k.lower() not in ("content-encoding", "transfer-encoding")},
                )
            finally:
                slot.active_requests -= 1
                elapsed = time.time() - start
                logger.info("Request to slot %d completed in %.3fs (active=%d)", slot.index, elapsed, slot.active_requests)

    async def _route_to_external(self, model_id: str, meta: Dict[str, Any], request: Request) -> Response:
        """Forward a request to an external HTTP endpoint (e.g., Strata).

        This path does not manage llama-server slots; it is used for models whose
        registry entry has an "external_endpoint" field.
        """
        endpoint = meta["external_endpoint"].rstrip("/")
        path = request.url.path
        if request.url.query:
            path += "?" + request.url.query
        target = f"{endpoint}{path}"
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host",)}
        method = request.method

        is_stream = False
        try:
            data = json.loads(body)
            is_stream = data.get("stream", False)
        except Exception:
            pass

        try:
            if is_stream:
                upstream_req = self.http_client.build_request(
                    method=method, url=target, headers=headers, content=body
                )
                resp = await self.http_client.send(upstream_req, stream=True)
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    await resp.aclose()
                    return Response(
                        content=error_body,
                        status_code=resp.status_code,
                        headers={"content-type": resp.headers.get("content-type", "application/json")},
                    )

                async def stream_response() -> AsyncIterator[bytes]:
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    except Exception as e:
                        logger.warning("External streaming request for %s aborted: %s", model_id, e)
                    finally:
                        await resp.aclose()
                        elapsed = time.time() - start
                        logger.info("External streaming request for %s completed in %.3fs", model_id, elapsed)

                start = time.time()
                return StreamingResponse(stream_response(), media_type="text/plain")
            else:
                start = time.time()
                resp = await self.http_client.request(method=method, url=target, headers=headers, content=body)
                elapsed = time.time() - start
                logger.info("External request for %s completed in %.3fs (status=%d)", model_id, elapsed, resp.status_code)
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers={k: v for k, v in resp.headers.items() if k.lower() not in ("content-encoding", "transfer-encoding")},
                )
        except httpx.ConnectError as exc:
            logger.warning("External endpoint %s for model %s unreachable: %s", endpoint, model_id, exc)
            raise HTTPException(
                status_code=503,
                detail=f"External endpoint for model '{model_id}' is unreachable; is Strata running? ({exc})",
            ) from exc
        except Exception as exc:
            logger.exception("External request to %s for model %s failed", endpoint, model_id)
            raise HTTPException(status_code=502, detail=f"External request failed: {exc}") from exc

    async def handle_chat_completions(self, request: Request) -> Response:
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        model_id = data.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model' field")
        meta = self.registry.get(model_id)
        if not meta:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not available in this orchestrator")

        # External endpoint routing (e.g., Strata). Does not occupy a GPU slot.
        if meta.get("external_endpoint"):
            return await self._route_to_external(model_id, meta, request)

        slot = await self._ensure_model_loaded(model_id)
        return await self._forward(slot, request)

    async def handle_completions(self, request: Request) -> Response:
        return await self.handle_chat_completions(request)

    async def list_models(self) -> Dict[str, Any]:
        models = []
        for mid, meta in self.registry.models.items():
            entry = {
                "id": mid,
                "object": "model",
                "owned_by": os.environ.get("ORCHESTRATOR_OWNER", "orchestrator"),
                "family": meta.get("family"),
                "attachment": meta.get("attachment"),
                "tool_call": meta.get("tool_call"),
            }
            if meta.get("external_endpoint"):
                entry["external"] = True
                entry["external_endpoint"] = meta["external_endpoint"]
            models.append(entry)
        return {"object": "list", "data": models}

    async def health(self) -> Dict[str, Any]:
        # Health reflects whether llama-server slots are ready, not whether
        # they match the boot default (models are swapped dynamically).
        # A slot that was intentionally stopped by /admin/slot/{i}/stop is
        # reported as ok=False.  By default the overall status is "ok" only
        # when every slot is ready; set ORCHESTRATOR_HEALTH_LENIENT=1 to treat
        # any ready slot as "ok" (useful when one slot is intentionally stopped
        # for single-GPU image-gen coexistence).
        ready_slots = [s for s in self.slots if s.state == "ready"]
        lenient = os.environ.get("ORCHESTRATOR_HEALTH_LENIENT", "").lower() in ("1", "true", "yes")
        status = []
        for slot in self.slots:
            status.append({
                "slot": slot.index,
                "gpu": slot.gpu,
                "state": slot.state,
                "loaded_model": slot.loaded_model,
                "active_requests": slot.active_requests,
                "queue_depth": slot.queue.qsize(),
                "ok": slot.state == "ready",
            })
        if ready_slots:
            if len(ready_slots) == len(self.slots):
                overall = "ok"
            else:
                overall = "ok" if lenient else "degraded"
        else:
            overall = "degraded"
        mode = get_node_mode()
        return {
            "status": overall,
            "all_ready": len(ready_slots) == len(self.slots),
            "node_mode": mode,
            "imagegen_active": mode in BUSY_MODES,
            "slots": status,
        }

    # --- Admin slot control ---

    async def stop_slot(self, slot_index: int) -> Dict[str, Any]:
        if slot_index < 0 or slot_index >= len(self.slots):
            raise HTTPException(status_code=404, detail=f"Slot {slot_index} does not exist")
        slot = self.slots[slot_index]
        async with slot.lock:
            await self._kill_slot(slot)
            await wait_for_vram_drain(slot.gpu, timeout=VRAM_DRAIN_TIMEOUT)
            slot.state = "stopped"
            slot.loaded_model = None
        logger.info("Slot %d stopped by admin", slot_index)
        return {"slot": slot_index, "state": slot.state, "gpu": slot.gpu}

    async def start_slot(self, slot_index: int) -> Dict[str, Any]:
        if slot_index < 0 or slot_index >= len(self.slots):
            raise HTTPException(status_code=404, detail=f"Slot {slot_index} does not exist")
        self.registry.refresh()
        slot = self.slots[slot_index]
        defaults = self.registry.default_boot
        if len(defaults) <= slot_index:
            default_model = defaults[0] if defaults else None
        else:
            default_model = defaults[slot_index]
        if not default_model:
            raise HTTPException(status_code=503, detail=f"No default model configured for slot {slot_index}")
        ok = await self._start_model(slot, default_model)
        if not ok:
            raise HTTPException(status_code=503, detail=f"Failed to start slot {slot_index} with {default_model}")
        logger.info("Slot %d started by admin with %s", slot_index, default_model)
        return {"slot": slot_index, "state": slot.state, "loaded_model": slot.loaded_model}

    async def slot_status(self, slot_index: int) -> Dict[str, Any]:
        if slot_index < 0 or slot_index >= len(self.slots):
            raise HTTPException(status_code=404, detail=f"Slot {slot_index} does not exist")
        slot = self.slots[slot_index]
        return {
            "slot": slot.index,
            "gpu": slot.gpu,
            "state": slot.state,
            "loaded_model": slot.loaded_model,
            "active_requests": slot.active_requests,
            "queue_depth": slot.queue.qsize(),
        }

    async def shutdown(self) -> None:
        logger.info("Shutting down orchestrator...")
        self.shutting_down = True
        # Stop accepting new work is handled by app lifecycle; here we drain.
        deadline = time.time() + GRACEFUL_SHUTDOWN_SECONDS
        while time.time() < deadline and any(s.active_requests > 0 for s in self.slots):
            await asyncio.sleep(0.5)
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await asyncio.gather(*(self._kill_slot(s) for s in self.slots), return_exceptions=True)
        try:
            self.save_state()
        except Exception:
            logger.exception("Failed to save state")
        await self.http_client.aclose()
        logger.info("Shutdown complete")

    def save_state(self) -> None:
        state = {
            "slots": [
                {
                    "index": s.index,
                    "gpu": s.gpu,
                    "loaded_model": s.loaded_model,
                    "state": s.state,
                    "active_requests": s.active_requests,
                }
                for s in self.slots
            ]
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="orchestrator", version="1.0.0")
orchestrator: Orchestrator


@app.on_event("startup")
async def startup_event():
    global orchestrator
    orchestrator = Orchestrator()
    await orchestrator.start_workers()
    # Run model loading in the background so uvicorn can bind the port immediately.
    asyncio.create_task(orchestrator.boot_background())


@app.on_event("shutdown")
async def shutdown_event():
    await orchestrator.shutdown()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await orchestrator.handle_chat_completions(request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await orchestrator.handle_completions(request)


@app.get("/v1/models")
async def models():
    return await orchestrator.list_models()


@app.get("/health")
async def health():
    return await orchestrator.health()


@app.post("/admin/slot/{slot_index}/stop")
async def admin_stop_slot(slot_index: int):
    return await orchestrator.stop_slot(slot_index)


@app.post("/admin/slot/{slot_index}/start")
async def admin_start_slot(slot_index: int):
    return await orchestrator.start_slot(slot_index)


@app.get("/admin/slot/{slot_index}")
async def admin_get_slot(slot_index: int):
    return await orchestrator.slot_status(slot_index)


# -----------------------------------------------------------------------------
# CLI / systemd entry point
# -----------------------------------------------------------------------------
def main() -> None:
    import uvicorn
    logger.info("Starting orchestrator on 0.0.0.0:%d", ORCHESTRATOR_PORT)
    uvicorn.run(app, host="0.0.0.0", port=ORCHESTRATOR_PORT, log_level="info", loop="asyncio")


if __name__ == "__main__":
    main()
