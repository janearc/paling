import subprocess
import sys
from pathlib import Path
from typing import Optional, List

def run_fuse(
    model: str,
    adapter_path: str = "adapters",
    save_path: str = "fused_model",
    dequantize: bool = False,
    export_gguf: bool = False,
    gguf_path: Optional[str] = None,
    extra_args: Optional[List[str]] = None
) -> int:
    """
    Invokes `mlx_lm fuse` via subprocess to merge LoRA adapter weights with the base model.
    """
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model,
        "--adapter-path", adapter_path,
        "--save-path", save_path
    ]

    if dequantize:
        cmd.append("--dequantize")
        
    if export_gguf:
        cmd.append("--export-gguf")
        if gguf_path:
            cmd.extend(["--gguf-path", gguf_path])

    if extra_args:
        cmd.extend(extra_args)

    print("=" * 60)
    print("Fusing LoRA Adapters into Base Model")
    print(f"  Base Model:   {model}")
    print(f"  Adapter Path: {adapter_path}")
    print(f"  Save Path:    {save_path}")
    print(f"  Dequantize:   {dequantize}")
    print(f"  Export GGUF:  {export_gguf}")
    if export_gguf and gguf_path:
        print(f"  GGUF Path:    {gguf_path}")
    print("=" * 60)
    print(f"Running command: {' '.join(cmd)}\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end="", flush=True)
                
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        print("\nFusing process interrupted.")
        if 'process' in locals():
            process.terminate()
            process.wait()
        return 130
    except Exception as e:
        print(f"Error running fusing process: {e}")
        return 1
