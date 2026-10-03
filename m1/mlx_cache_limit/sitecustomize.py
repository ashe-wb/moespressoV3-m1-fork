"""Cap MLX's buffer cache in a process started through m1/moespresso-m1.

With this directory on PYTHONPATH and MLX_CACHE_LIMIT_GB set, the limit is
applied immediately after the process imports mlx.core, and nothing is imported
early. MLX keeps freed GPU buffers in a cache that the MoEspresso memory planner
does not count. On a host whose GPU driver keeps only part of RAM resident, that
cache can push the process past the resident budget and stall decode.
"""
import importlib.abc
import importlib.machinery
import os
import sys

_LIMIT = os.environ.get("MLX_CACHE_LIMIT_GB")


class _AfterImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "mlx.core":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        original = loader.exec_module

        def exec_module(module):
            original(module)
            try:
                module.set_cache_limit(int(float(_LIMIT) * 2**30))
                print(f"[moespresso-m1] MLX buffer cache limited to {_LIMIT} GB", file=sys.stderr, flush=True)
            except Exception as e:  # A failed limit must not stop the server.
                print(f"[moespresso-m1] could not limit the MLX buffer cache: {e!r}", file=sys.stderr, flush=True)

        loader.exec_module = exec_module
        return spec


if _LIMIT:
    sys.meta_path.insert(0, _AfterImport())
