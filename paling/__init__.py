# paling package
__version__ = "0.1.0"


# lazy export: importing a leaf module (e.g. `paling.launchagent`, `paling.daemon`)
# must not drag in the whole MLX/torch chain that `paling.cli` pulls. `main` is
# still importable as `from paling import main` via PEP 562 -- it just resolves on
# first access instead of at package import time.
def __getattr__(name):
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module 'paling' has no attribute '{name}'")
