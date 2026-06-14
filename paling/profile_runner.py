import json
import logging
from pathlib import Path
from typing import Optional, Any
from wonderlib.profiling import profile_sigil
from wonderlib.git_stats import get_git_stats

logger = logging.getLogger(__name__)

def profile_single_file(
    file_path: Path,
    output_dir: Path,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    model_path: Optional[str] = None,
    include_git: bool = True
) -> bool:
    """
    Profiles a single markdown file and saves its taxonometry JSON.
    """
    try:
        if not file_path.exists():
            logger.info(f"❌ Error: Sigil file '{file_path}' does not exist.")
            return False
            
        # Load MLX model if requested and not already passed
        if not model and model_path:
            try:
                logger.info(f"Loading MLX model from '{model_path}' for rare term extraction...")
                from mlx_lm import load
                model, tokenizer = load(model_path)
            except Exception as e:
                logger.info(f"⚠️ Warning: Failed to load MLX model '{model_path}': {e}")
                logger.info("Falling back to fast lexical heuristics.")
                
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        sigil_stem = file_path.stem
        logger.info(f"📊 Profiling sigil '{sigil_stem}' ({file_path.name})...")
        
        # Calculate taxonometry metrics (falls back to lexical heuristics if model/tokenizer are None)
        profiled = profile_sigil(
            text=content,
            title=sigil_stem,
            model=model,
            tokenizer=tokenizer
        )
        
        profiled.filename = str(file_path.resolve())
        
        # Load git statistics if requested
        if include_git:
            try:
                git_stats = get_git_stats(str(file_path))
                profiled.git_stats = git_stats
            except Exception as e:
                logger.info(f"⚠️ Warning: Failed to extract Git statistics for {file_path.name}: {e}")
                
        # Write to JSON
        output_dir.mkdir(parents=True, exist_ok=True)
        out_json_path = output_dir / f"{sigil_stem}-taxonometry.json"
        
        with open(out_json_path, "w", encoding="utf-8") as f:
            f.write(profiled.model_dump_json(indent=2))
            
        logger.info(f"✅ Saved taxonometry profile to: {out_json_path}")
        if profiled.benchmark:
            profiled.benchmark.report()
            
        return True
    except Exception as e:
        logger.info(f"❌ Failed to profile '{file_path.name}': {e}")
        import traceback
        traceback.print_exc()
        return False

def profile_directory(
    input_dir: Path,
    output_dir: Path,
    model_path: Optional[str] = None,
    include_git: bool = True,
    fix_only: bool = False
):
    """
    Profiles all markdown files in a directory recursively.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory '{input_dir}' is not valid.")
        
    sigil_files = list(input_dir.rglob("*.md"))
    if not sigil_files:
        logger.info(f"No markdown files found in '{input_dir}'.")
        return
        
    logger.info(f"Discovered {len(sigil_files)} sigil files under '{input_dir}'.")
    
    # Filter files if fix_only is specified
    files_to_profile = []
    for sfile in sigil_files:
        if fix_only:
            out_file = output_dir / f"{sfile.stem}-taxonometry.json"
            if out_file.exists():
                # Extant signature, skip
                continue
        files_to_profile.append(sfile)
        
    if not files_to_profile:
        logger.info("All sigil signatures are already up-to-date.")
        return
        
    logger.info(f"Preparing to profile {len(files_to_profile)} files...")
    
    # Load MLX model if model_path is specified, otherwise run model-free
    model = None
    tokenizer = None
    if model_path:
        try:
            logger.info(f"Loading MLX model from '{model_path}' for rare term extraction...")
            from mlx_lm import load
            model, tokenizer = load(model_path)
        except Exception as e:
            logger.info(f"⚠️ Warning: Failed to load MLX model '{model_path}': {e}")
            logger.info("Falling back to fast lexical heuristics.")
            
    success_count = 0
    for sfile in files_to_profile:
        success = profile_single_file(
            file_path=sfile,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            include_git=include_git
        )
        if success:
            success_count += 1
            
    logger.info(f"Done. Successfully profiled {success_count}/{len(files_to_profile)} files.")
