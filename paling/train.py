import subprocess
import sys
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

def run_training(
    model: str,
    data_dir: str,
    adapter_path: str = "adapters",
    iters: int = 1000,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    max_seq_length: int = 2048,
    fine_tune_type: str = "lora",
    extra_args: Optional[List[str]] = None
) -> int:
    """Invokes `mlx_lm lora` via subprocess to run QLoRA / LoRA training.
    """
    # Verify dataset files exist
    data_path = Path(data_dir)
    train_file = data_path / "train.jsonl"
    valid_file = data_path / "valid.jsonl"
    
    if not train_file.exists() or not valid_file.exists():
        raise FileNotFoundError(
            f"Dataset files train.jsonl and valid.jsonl not found in '{data_dir}'. "
            f"Run the dataset preparation step first."
        )

    # Build the execution command
    # Use python3 -m mlx_lm lora as standard
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model,
        "--train",
        "--data", data_dir,
        "--adapter-path", adapter_path,
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--max-seq-length", str(max_seq_length),
        "--fine-tune-type", fine_tune_type,
        "--grad-checkpoint" # Save memory on M3 Max by checkpointing activations
    ]

    if extra_args:
        cmd.extend(extra_args)

    logger.info("=" * 60)
    logger.info("Starting MLX QLoRA Fine-tuning")
    logger.info(f"  Model:        {model}")
    logger.info(f"  Data Dir:     {data_dir}")
    logger.info(f"  Adapter Path: {adapter_path}")
    logger.info(f"  Iterations:   {iters}")
    logger.info(f"  Batch Size:   {batch_size}")
    logger.info(f"  Learning Rate:{learning_rate}")
    logger.info(f"  Type:         {fine_tune_type.upper()}")
    logger.info("=" * 60)
    logger.info(f"Running command: {' '.join(cmd)}\n")

    # Start the training process and stream output to console
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
                sys.stdout.write(str(line))
                sys.stdout.flush()
                
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
        if 'process' in locals():
            process.terminate()
            process.wait()
        return 130
    except Exception as e:
        logger.info(f"Error running training process: {e}")
        return 1
