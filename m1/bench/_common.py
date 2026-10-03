"""Shared helpers for the m1 benchmark drivers."""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Importing mlx before MoEspresso skips the server's Qwen4 command-buffer
# defaults, so apply them here unless the environment already sets them.
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "50")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "200")

import mlx.core as mx  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def package_path() -> Path:
    """Return the package from MOESPRESSO_M1_PACKAGE, or the README's download path."""
    return Path(os.environ.get("MOESPRESSO_M1_PACKAGE", "models/qwen3.8-flash-next")).expanduser()


def start_memory_guard(tag: str) -> None:
    """Cap MLX's buffer cache and exit when macOS reports under 8% free memory.

    Large chunks otherwise push the host into memory compression, which stalls
    decode and can leave the host unresponsive.
    """
    mx.set_cache_limit(1 << 30)

    def guard():
        while True:
            time.sleep(2)
            out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
            free = int(out.strip().splitlines()[-1].split(":")[-1].strip().rstrip("%"))
            if free < 8:
                print(f"{tag} abort: memory free {free}%", file=sys.stderr, flush=True)
                os._exit(3)

    threading.Thread(target=guard, daemon=True).start()


def chat_ids(tokenizer, prompt: str) -> list[int]:
    """Render one user turn with thinking off and return its token ids."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False,
        enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def routing_kwargs(policy: str) -> dict:
    """Map POLICY=off|<factor>,<protected> to load_served_model routing arguments."""
    if policy == "off":
        return {"cache_routing": "off"}
    import moespresso.runtime.qwen4.cache_routing_config as crc

    crc.MAX_CACHE_FACTOR = 1e9
    factor, protected = policy.split(",")
    return {"cache_routing": "prefer-resident", "cache_routing_factor": float(factor),
            "cache_routing_protected_routes": int(protected)}
