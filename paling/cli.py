#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ensure the parent directory is in python path to import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paling.dataset import build_datasets, DEFAULT_SYSTEM_PROMPT
from paling.painter import run_painter
from paling.train import run_training
from paling.inference import run_interactive_chat
from paling.fuse import run_fuse
from paling.profile_runner import profile_single_file, profile_directory

def main():
    parser = argparse.ArgumentParser(
        description="Paling CLI: QLoRA Fine-tuning and Inference tool for Markdown Knowledge Bases on Apple Silicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")
    
    # Subcommand: create
    parser_create = subparsers.add_parser(
        "create",
        help="Create resources (e.g. bento)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    create_subparsers = parser_create.add_subparsers(dest="create_command", help="Resource to create")
    
    parser_bento = create_subparsers.add_parser(
        "bento",
        help="Scaffold a new Paling Bento directory locally",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_bento.add_argument(
        "--name", "-n",
        help="Explicit name of the Bento"
    )
    parser_bento.add_argument(
        "--new",
        action="store_true",
        help="Generate a random UUID for the Bento name"
    )
    parser_bento.add_argument(
        "--dir", "-d",
        default=str(Path.home() / "var" / "paling" / "bentos"),
        help="The directory to create the Bento in"
    )
    parser_bento.add_argument(
        "--type", "-t",
        choices=["logs", "corpus", "unprocessed"],
        default="unprocessed",
        help="The archetype classification for the data"
    )

    # Subcommand: submit
    parser_submit = subparsers.add_parser(
        "submit",
        help="Submit a local Bento directory to the Paling daemon processing spool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_submit.add_argument(
        "path",
        help="Path to the local Bento directory to submit"
    )
    
    # Subcommand: prepare
    parser_prep = subparsers.add_parser(
        "prepare",
        help="Process markdown documentation into JSONL training/validation datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_prep.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing markdown files (.md) to process"
    )
    parser_prep.add_argument(
        "--output-dir", "-o",
        default="data",
        help="Output directory to write train.jsonl and valid.jsonl"
    )
    parser_prep.add_argument(
        "--mode", "-m",
        choices=["sections", "raw_text", "qa_pairs"],
        default="sections",
        help="Preparation mode: 'sections' (split by headers to build QA pairs), 'raw_text' (sliding window chunks), 'qa_pairs' (full file QA)"
    )
    parser_prep.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=500,
        help="Chunk size (in words, or tokens if --model-path is provided) for raw_text mode"
    )
    parser_prep.add_argument(
        "--overlap", "-p",
        type=int,
        default=50,
        help="Overlap size (in words/tokens) for raw_text mode"
    )
    parser_prep.add_argument(
        "--val-split", "-v",
        type=float,
        default=0.1,
        help="Fraction of the dataset to reserve for validation"
    )
    parser_prep.add_argument(
        "--model-path",
        help="Optional model path or HF repo ID to load the tokenizer for precise token-based chunking in raw_text mode"
    )
    parser_prep.add_argument(
        "--rlhf-dir",
        help="Optional path to directory containing RLHF JSON review files (*-review.json) to incorporate"
    )
    parser_prep.add_argument(
        "--taxonometry-dir",
        help="Optional path to directory containing taxonometry JSON files (*-taxonometry.json) to incorporate"
    )
    parser_prep.add_argument(
        "--system-prompt", "-s",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for instruction modes ('sections' and 'qa_pairs')"
    )
    parser_prep.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting train/validation datasets"
    )

    # Subcommand: train
    parser_train = subparsers.add_parser(
        "train",
        help="Run LoRA/QLoRA training on Apple Silicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_train.add_argument(
        "--model", "-m",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Hugging Face model ID or path to a local MLX model directory"
    )
    parser_train.add_argument(
        "--data", "-d",
        default="data",
        help="Directory containing train.jsonl and valid.jsonl files"
    )
    parser_train.add_argument(
        "--adapter-path", "-a",
        default="adapters",
        help="Path where fine-tuned LoRA weights and configs should be saved"
    )
    parser_train.add_argument(
        "--iters", "-i",
        type=int,
        default=1000,
        help="Number of training iterations"
    )
    parser_train.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size (number of sequences processed simultaneously)"
    )
    parser_train.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for Adam optimizer"
    )
    parser_train.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length of training input"
    )
    parser_train.add_argument(
        "--fine-tune-type",
        choices=["lora", "dora", "full"],
        default="lora",
        help="Fine-tuning mechanism: lora (LoRA), dora (DoRA), full (Full parameters)"
    )

    # Subcommand: chat
    parser_chat = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session with the base or fine-tuned model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_chat.add_argument(
        "--model", "-m",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Hugging Face model ID or path to a local model directory"
    )
    parser_chat.add_argument(
        "--adapter-path", "-a",
        default="adapters",
        help="Path to the directory containing fine-tuned LoRA adapter weights (optional)"
    )
    parser_chat.add_argument(
        "--system-prompt", "-s",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to structure model responses"
    )

    # Subcommand: fuse
    parser_fuse = subparsers.add_parser(
        "fuse",
        help="Fuse LoRA adapters back into the base model weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_fuse.add_argument(
        "--model", "-m",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Hugging Face model ID or path to local model directory"
    )
    parser_fuse.add_argument(
        "--adapter-path", "-a",
        default="adapters",
        help="Path to trained adapter weights directory"
    )
    parser_fuse.add_argument(
        "--save-path", "-s",
        default="fused_model",
        help="Path where the fused model weights should be saved"
    )
    parser_fuse.add_argument(
        "--dequantize",
        action="store_true",
        help="Export a dequantized version of the model"
    )
    parser_fuse.add_argument(
        "--export-gguf",
        action="store_true",
        help="Export fused weights in GGUF format"
    )
    parser_fuse.add_argument(
        "--gguf-path",
        help="Filename to save the exported GGUF model (defaults to ggml-model-f16.gguf under save-path)"
    )
    # Subcommand: paint
    parser_paint = subparsers.add_parser(
        "paint",
        help="Run Painter LLM to generate Anchor Banchan (high-reward hallucinations)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_paint.add_argument(
        "--target-model",
        "-t",
        required=True,
        help="Identifier (HF repo ID or local path) of the target model to provoke"
    )
    parser_paint.add_argument(
        "--painter-model",
        "-p",
        help="Path to a small painter model (optional). If omitted, a mock stub is used."
    )
    parser_paint.add_argument(
        "--bento-dir",
        "-b",
        default="bento/default",
        help="Bento directory where Anchor Banchan JSON will be stored"
    )
    parser_paint.add_argument(
        "--steps",
        "-s",
        type=int,
        default=20,
        help="Number of paint interaction steps to run"
    )

    parser_paint.add_argument(
        "--reward-threshold",
        "-r",
        type=float,
        default=0.6,
        help="Score threshold (0-1) for saving an interaction as high-reward"
    )

    # Subcommand: serve
    parser_serve = subparsers.add_parser(
        "serve",
        help="Run the paling API daemon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_serve.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Port to run the daemon on"
    )

    # Subcommand: profile
    parser_prof = subparsers.add_parser(
        "profile",
        help="Generate taxonometric profile of documents using Zipf average, part-of-speech complexity, and rare term extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_prof.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a markdown file (.md) or a directory of markdown files to profile"
    )
    parser_prof.add_argument(
        "--output-dir", "-o",
        default="data/taxonometry/profiles",
        help="Directory where the generated taxonometry JSON profiles should be saved"
    )
    parser_prof.add_argument(
        "--model-path",
        default=None,
        help="Optional Hugging Face model ID or path to local MLX model directory for LLM-based rare term extraction (runs offline lexical heuristic by default)"
    )
    parser_prof.add_argument(
        "--no-git",
        action="store_true",
        help="Skip extracting Git edit statistics for the files"
    )
    parser_prof.add_argument(
        "--fix-only",
        action="store_true",
        help="Only profile files that do not already have a signature JSON in the output directory"
    )

    # Allow passing extra args to train and fuse for native mlx-lm flags
    args, unknown_args = parser.parse_known_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "create":
        if args.create_command == "bento":
            import uuid
            
            bento_name = args.name
            if args.new:
                bento_name = str(uuid.uuid4())
                
            if not bento_name:
                logger.error("Error: You must specify either --name <string> or --new")
                sys.exit(1)
                
            # Scaffold a bento locally
            base_path = Path(args.dir).resolve() / bento_name
        if base_path.exists():
            logger.error(f"Directory '{base_path}' already exists.")
            sys.exit(1)
        
        directories = [
            "raw_data", "schema", "adapters", "preflight",
            "taxonometry", "anchors/owner", "anchors/paling",
            "acceptance", "output"
        ]
        logger.info(f"Scaffolding Bento: {bento_name} (Type: {args.type})...")
        for dir_name in directories:
            (base_path / dir_name).mkdir(parents=True, exist_ok=True)
            
        schema_path = base_path / "schema" / "schema.json"
        schema_content = f'''{{
  "archetype": "{args.type}",
  "routing": {{
    "gap_generation": "flan-t5-large",
    "summarization": "mistral"
  }}
}}'''
        with open(schema_path, "w") as f:
            f.write(schema_content)
            
        logger.info(f"Success! Bento scaffolded at: {base_path}")
        sys.exit(0)
        
    elif args.command == "submit":
        import subprocess
        source = Path(args.path).resolve()
        if not source.exists() or not source.is_dir():
            logger.error(f"Error: '{source}' is not a valid directory.")
            sys.exit(1)
            
        target_dir = str(Path.home() / "var" / "paling" / "bentos") + "/"
        logger.info(f"Submitting '{source.name}' to Paling processing queue at {target_dir}...")
        cmd = ["rsync", "-a", "--info=progress2", str(source), target_dir]
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully submitted '{source.name}'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during rsync submission: {e}")
            sys.exit(1)
        sys.exit(0)

    elif args.command == "prepare":
        try:
            build_datasets(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                mode=args.mode,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                val_split=args.val_split,
                seed=args.seed,
                model_path=args.model_path,
                system_prompt=args.system_prompt,
                rlhf_dir=args.rlhf_dir,
                taxonometry_dir=args.taxonometry_dir
            )
        except Exception as e:
            logger.error(f"Error during preparation: {e}")
            sys.exit(1)
            
    elif args.command == "train":
        # Pass unknown arguments as extra args to training
        exit_code = run_training(
            model=args.model,
            data_dir=args.data,
            adapter_path=args.adapter_path,
            iters=args.iters,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            fine_tune_type=args.fine_tune_type,
            extra_args=unknown_args
        )
        sys.exit(exit_code)
        
    elif args.command == "chat":
        # If adapters path directory doesn't have an adapter_config.json, assume no LoRA loading
        adapter_path = args.adapter_path
        if adapter_path:
            p = Path(adapter_path)
            if not (p / "adapter_config.json").exists() and not (p / "adapters.safetensors").exists():
                logger.info(f"LoRA adapter files not found in '{adapter_path}'. Running base model only.")
                adapter_path = None
                
        run_interactive_chat(
            model_path=args.model,
            adapter_path=adapter_path,
            system_prompt=args.system_prompt
        )
        
    elif args.command == "fuse":
        exit_code = run_fuse(
            model=args.model,
            adapter_path=args.adapter_path,
            save_path=args.save_path,
            dequantize=args.dequantize,
            export_gguf=args.export_gguf,
            gguf_path=args.gguf_path,
            extra_args=unknown_args
        )
        sys.exit(exit_code)
        
    elif args.command == "profile":
        input_path = Path(args.input)
        output_path = Path(args.output_dir)
        include_git = not args.no_git
        
        if input_path.is_file():
            profile_single_file(
                file_path=input_path,
                output_dir=output_path,
                model_path=args.model_path,
                include_git=include_git
            )
        elif input_path.is_dir():
            profile_directory(
                input_dir=input_path,
                output_dir=output_path,
                model_path=args.model_path,
                include_git=include_git,
                fix_only=args.fix_only
            )
        else:
            logger.error(f"Error: Input path '{args.input}' does not exist or is neither a file nor a directory.")
            sys.exit(1)

    elif args.command == "paint":
        # Run the Painter LLM pipeline with reward threshold
        run_painter(
            target_model=args.target_model,
            painter_model=args.painter_model,
            bento_dir=args.bento_dir,
            num_interactions=args.steps,
            reward_threshold=args.reward_threshold,
        )
        sys.exit(0)
        
    elif args.command == "serve":
        from paling.daemon import serve
        serve(port=args.port)
        sys.exit(0)

if __name__ == "__main__":
    main()
