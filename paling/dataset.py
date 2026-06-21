import json
import logging
import os
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# A default system prompt for knowledge injection
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the user's personal notes and "
    "documentation. Use this knowledge to answer questions accurately and concisely."
)

def parse_markdown_to_sections(text: str, filepath: Path) -> List[Dict[str, Any]]:
    """
    Parses a markdown file into hierarchical sections based on header markers.
    """
    lines = text.split('\n')
    sections = []
    
    # Track the current state of headers
    current_headers = {i: "" for i in range(1, 7)}
    current_content = []
    current_level = 0
    current_header_text = "Introduction"
    
    header_pattern = re.compile(r'^(#{1,6})\s+(.*)$')
    
    # Process line-by-line
    for line in lines:
        match = header_pattern.match(line)
        if match:
            # Save the previous section if it has content or isn't the empty intro
            content_str = "\n".join(current_content).strip()
            if content_str:
                headers_path = [
                    current_headers[i]
                    for i in range(1, current_level + 1)
                    if current_headers[i]
                ]
                sections.append({
                    "file_path": str(filepath),
                    "file_name": filepath.name,
                    "title": filepath.stem,
                    "header": current_header_text,
                    "headers_path": headers_path,
                    "content": content_str
                })
            
            # Start new section
            level = len(match.group(1))
            header_title = match.group(2).strip()
            
            current_headers[level] = header_title
            # Clear deeper header paths
            for i in range(level + 1, 7):
                current_headers[i] = ""
                
            current_level = level
            current_header_text = header_title
            current_content = []
        else:
            current_content.append(line)
            
    # Save the last section
    content_str = "\n".join(current_content).strip()
    if content_str:
        headers_path = [
            current_headers[i]
            for i in range(1, current_level + 1)
            if current_headers[i]
        ]
        sections.append({
            "file_path": str(filepath),
            "file_name": filepath.name,
            "title": filepath.stem,
            "header": current_header_text,
            "headers_path": headers_path,
            "content": content_str
        })
        
    return sections

def chunk_text_by_words(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks based on word counts.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks

def chunk_text_by_tokens(
    text: str, tokenizer: Any, chunk_size: int = 1024, overlap: int = 128
) -> List[str]:
    """
    Splits text into overlapping chunks using a tokenizer.
    """
    tokens = tokenizer.encode(text)
    if not tokens:
        return []
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    return chunks

def extract_markdown_files(input_dir: Path, exclude_patterns: List[str]) -> List[Path]:
    """
    Finds all markdown files recursively, ignoring excluded directories.
    """
    markdown_files = []
    for root, dirs, files in os.walk(input_dir):
        # Filter directories in-place to avoid traversing excluded ones
        dirs[:] = [d for d in dirs if not any(re.search(pat, d) for pat in exclude_patterns)]
        
        for file in files:
            if file.endswith('.md') and not file.startswith('.'):
                file_path = Path(root) / file
                if not any(re.search(pat, str(file_path)) for pat in exclude_patterns):
                    markdown_files.append(file_path)
    return markdown_files

def parse_rlhf_directory(rlhf_dir: Path) -> List[Dict[str, str]]:
    """
    Parses RLHF reviews from a directory, collecting approved QA pairs.
    """
    rlhf_data = []
    if not rlhf_dir.exists():
        logger.info(f"Warning: RLHF directory '{rlhf_dir}' does not exist. Skipping.")
        return rlhf_data
        
    for file in rlhf_dir.glob("**/*-review.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            for q_entry in data.get("questions", []):
                if not q_entry.get("approved", False):
                    continue
                    
                question = q_entry.get("question", "").strip()
                if not question:
                    continue
                    
                response = q_entry.get("synthesis_answer", "").strip()
                if not response:
                    answers = q_entry.get("answers", [])
                    if answers:
                        first_ans = answers[0]
                        if isinstance(first_ans, dict):
                            response = first_ans.get("answer", "").strip()
                        elif isinstance(first_ans, str):
                            response = first_ans.strip()
                            
                if response:
                    rlhf_data.append({
                        "question": question,
                        "response": response,
                        "source": file.name
                    })
        except Exception as e:
            logger.info(f"Warning: Failed to parse RLHF file '{file}': {e}")
            
    logger.info(f"Parsed {len(rlhf_data)} approved QA pairs from RLHF reviews.")
    return rlhf_data

def parse_taxonometry_directory(tax_dir: Path) -> List[Dict[str, Any]]:
    """
    Parses taxonometry metrics from a directory.
    """
    tax_data = []
    if not tax_dir.exists():
        logger.info(f"Warning: Taxonometry directory '{tax_dir}' does not exist. Skipping.")
        return tax_data
        
    for file in tax_dir.glob("**/*-taxonometry.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            title = data.get("title", file.stem.replace("-taxonometry", "")).strip()
            zipf_avg = data.get("zipf_avg", 0.0)
            rarity_pos = data.get("rarity_pos", 0.0)
            rare_terms = [t for t in data.get("rare_terms", []) if isinstance(t, str)]
            
            tax_data.append({
                "title": title,
                "zipf_avg": zipf_avg,
                "rarity_pos": rarity_pos,
                "rare_terms": rare_terms,
                "source": file.name
            })
        except Exception as e:
            logger.info(f"Warning: Failed to parse Taxonometry file '{file}': {e}")
            
    logger.info(f"Parsed {len(tax_data)} Taxonometry profile definitions.")
    return tax_data

def _write_split(
    records: List[Dict[str, Any]],
    output_path: Path,
    val_split: float,
    seed: int,
    label: str,
) -> Tuple[int, int]:
    """Shuffle one record list, split into train/valid, write both into out/<label>/."""
    # Nothing to write.
    if not records:
        return 0, 0

    # Shuffle with a fixed seed so the same inputs always split the same way
    # (reproducible datasets).
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    # Cut the list into train (the front) and valid (the tail). e.g. val_split
    # of 0.1 keeps 90% for training.
    split_idx = int(len(shuffled) * (1 - val_split))
    train_records = shuffled[:split_idx]
    val_records = shuffled[split_idx:]
    # Edge case: if rounding left valid empty but we have more than one record,
    # peel off the last one so valid is never empty when it could be non-empty.
    if not val_records and len(shuffled) > 1:
        train_records = shuffled[:-1]
        val_records = shuffled[-1:]

    # Each dataset gets its own subdir with the standard train.jsonl/valid.jsonl
    # names, so `paling train --data out/<label>` just works.
    split_dir = output_path / label
    split_dir.mkdir(parents=True, exist_ok=True)
    train_file = split_dir / "train.jsonl"
    valid_file = split_dir / "valid.jsonl"
    # Write one JSON object per line (JSONL). ensure_ascii=False keeps emoji and
    # other non-ascii readable instead of escaping them to \uXXXX.
    with open(train_file, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(valid_file, "w", encoding="utf-8") as f:
        for rec in val_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        f"  {label}: {len(train_records)} train / {len(val_records)} valid "
        f"-> {train_file}, {valid_file}"
    )
    return len(train_records), len(val_records)


def build_chatlog_datasets(
    input_dir: str,
    output_dir: str,
    val_split: float = 0.1,
    seed: int = 42,
    system_prompt: Optional[str] = None,
    skip_bad: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Read a dir of scraped chatlog JSON and write the CHARACTER + PAINTER datasets.

    This is stage 2 of the chatlog pipeline (see docs/pipeline/chatlog-ingest.md).
    Reads every *.json file in input_dir, builds two datasets in their own subdirs
    of output_dir (character/ and painter/), writes a manifest, and returns
    {label: (n_train, n_valid)}. By default a single unparseable file aborts the
    whole build; pass skip_bad=True to skip bad files and keep going.
    """
    # Imported here (not at module top) to avoid a circular import.
    from paling import chatlog

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Accept either a single file or a directory of *.json files.
    if input_path.is_file():
        json_files = [input_path]
    else:
        json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No chatlog JSON files found in {input_dir}")

    logger.info(f"Found {len(json_files)} chatlog JSON file(s).")

    # Accumulators: the two datasets we're building, the files that failed to
    # parse, and a per-file count for the manifest.
    character_records: List[Dict[str, Any]] = []
    painter_records: List[Dict[str, Any]] = []

    failed: List[Dict[str, str]] = []
    per_file: List[Dict[str, Any]] = []

    for jf in json_files:
        # Parse one file into turns. Record the failure and move on rather than
        # crashing mid-loop -- we decide what to do about failures after.
        try:
            turns = chatlog.load_chatlog(jf)
        except Exception as e:
            logger.warning(f"Failed to parse chatlog {jf}: {e}")
            failed.append({"file": str(jf), "error": str(e)})
            continue

        # Build both datasets from the same turns and append to the running totals.
        char_recs = chatlog.character_records(turns, system_prompt)
        pair_recs = chatlog.painter_records(turns)
        character_records.extend(char_recs)
        painter_records.extend(pair_recs)

        # Note what this file contributed, for the manifest.
        entry = {
            "file": str(jf),
            "turns": len(turns),
            "character_records": len(char_recs),
            "painter_pairs": len(pair_recs),
        }
        per_file.append(entry)
        logger.info(
            f"  {jf.name}: {len(turns)} turns -> "
            f"{len(char_recs)} character records, {len(pair_recs)} painter pairs"
        )

    # Fail loud on bad inputs rather than train on partial data, unless the
    # caller explicitly opted into skip-and-warn.
    if failed and not skip_bad:
        names = "\n".join(f"  {f['file']}: {f['error']}" for f in failed)
        raise ValueError(
            f"{len(failed)} of {len(json_files)} chatlog file(s) failed to "
            f"parse; refusing to build datasets on partial data "
            f"(pass skip_bad=True to skip these):\n{names}"
        )

    # Nothing usable came out of any file -> there's no dataset to write.
    if not character_records and not painter_records:
        raise ValueError("No chatlog records generated. Check inputs.")

    logger.info("Chatlog dataset creation complete:")
    # Shuffle/split/write each dataset into its own subdir; keep the counts.
    results = {
        "character": _write_split(character_records, output_path, val_split, seed, "character"),
        "painter": _write_split(painter_records, output_path, val_split, seed, "painter"),
    }

    # Write a manifest next to the datasets recording exactly what went in, what
    # failed, and where each split landed -- so a build is always auditable.
    manifest = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "skip_bad": skip_bad,
        "inputs": per_file,
        "failed": failed,
        "datasets": {
            label: {
                "train": str(output_path / label / "train.jsonl"),
                "valid": str(output_path / label / "valid.jsonl"),
                "n_train": n_train,
                "n_valid": n_valid,
            }
            for label, (n_train, n_valid) in results.items()
        },
    }
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"  manifest -> {manifest_path}")

    return results


def build_datasets(
    input_dir: str,
    output_dir: str,
    mode: str = "sections",
    chunk_size: int = 500,
    overlap: int = 50,
    val_split: float = 0.1,
    seed: int = 42,
    model_path: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    exclude_patterns: Optional[List[str]] = None,
    rlhf_dir: Optional[str] = None,
    taxonometry_dir: Optional[str] = None
) -> Tuple[int, int]:
    """Build train/validation JSONL datasets from markdown, RLHF data, and taxonometry."""
    # The "chatlog" mode is a different beast: it reads scraped chatlog JSON
    # (not markdown) and writes two paired datasets via build_chatlog_datasets.
    # Hand it off, sum the per-dataset counts, and return the combined totals.
    if mode == "chatlog":
        results = build_chatlog_datasets(
            input_dir=input_dir,
            output_dir=output_dir,
            val_split=val_split,
            seed=seed,
            # The character side carries its own system prompt from the log; only
            # fall back to the explicit prompt if the caller overrode the default.
            system_prompt=None if system_prompt == DEFAULT_SYSTEM_PROMPT else system_prompt,
        )
        total_train = sum(t for t, _ in results.values())
        total_valid = sum(v for _, v in results.values())
        return total_train, total_valid

    if exclude_patterns is None:
        exclude_patterns = [r'\.git', r'\.venv', r'\.obsidian', r'node_modules']
        
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    md_files = extract_markdown_files(input_path, exclude_patterns)
    if not md_files:
        raise ValueError(f"No markdown files found in {input_dir}")
        
    logger.info(f"Found {len(md_files)} markdown files.")
    
    # Try loading tokenizer if model_path is provided and mode requires chunking
    tokenizer = None
    if model_path and mode == "raw_text":
        try:
            logger.info(f"Loading tokenizer from {model_path} for token-based chunking...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            logger.info(f"Failed to load tokenizer ({e}). Falling back to word-based chunking.")
            
    dataset_records = []
    
    # 1. Process Markdown documentation
    for md_file in md_files:
        try:
            # Get path relative to input_dir for clean references
            rel_path = md_file.relative_to(input_path)
            with open(md_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            if not content.strip():
                continue
                
            if mode == "sections":
                sections = parse_markdown_to_sections(content, rel_path)
                for sec in sections:
                    title = sec["title"]
                    header = sec["header"]
                    headers_path = sec["headers_path"]
                    sec_content = sec["content"]
                    
                    # Create prompt mapping hierarchy
                    path_str = " -> ".join(headers_path) if headers_path else header
                    prompt = (
                        f"In the document '{sec['file_name']}' under "
                        f"'{path_str}', what is written?"
                    )
                    
                    record = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": sec_content}
                        ]
                    }
                    dataset_records.append(record)
                    
            elif mode == "raw_text":
                # Chunk the raw file
                if tokenizer:
                    chunks = chunk_text_by_tokens(content, tokenizer, chunk_size, overlap)
                else:
                    chunks = chunk_text_by_words(content, chunk_size, overlap)
                    
                for idx, chunk in enumerate(chunks):
                    # Add document context at the top of the chunk to ground it
                    grounded_text = (
                        f"Document: {rel_path}\nChunk {idx+1}/{len(chunks)}\n---\n{chunk}"
                    )
                    dataset_records.append({"text": grounded_text})
                    
            elif mode == "qa_pairs":
                prompt = f"Retrieve the complete content of the note/document '{rel_path}'."
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": content.strip()}
                    ]
                }
                dataset_records.append(record)
                
        except Exception as e:
            logger.info(f"Error processing {md_file}: {e}")
            
    # 2. Incorporate RLHF QA Pairs
    if rlhf_dir:
        rlhf_path = Path(rlhf_dir)
        rlhf_pairs = parse_rlhf_directory(rlhf_path)
        for pair in rlhf_pairs:
            q = pair["question"]
            a = pair["response"]
            if mode in ["sections", "qa_pairs"]:
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a}
                    ]
                }
                dataset_records.append(record)
            elif mode == "raw_text":
                dataset_records.append({
                    "text": f"Question: {q}\nAnswer: {a}"
                })
                
    # 3. Incorporate Taxonometry Data
    if taxonometry_dir:
        tax_path = Path(taxonometry_dir)
        tax_profiles = parse_taxonometry_directory(tax_path)
        for profile in tax_profiles:
            title = profile["title"]
            zipf_avg = profile["zipf_avg"]
            rarity_pos = profile["rarity_pos"]
            rare_terms = profile["rare_terms"]
            
            if mode in ["sections", "qa_pairs"]:
                # Record 1: Profile overview
                prompt = f"What is the taxonometric profile of the document '{title}'?"
                response = (
                    f"The document '{title}' has a Zipf average of {zipf_avg:.3f} "
                    f"and rarity position of {rarity_pos:.5f}."
                )
                if rare_terms:
                    response += f" Its rare terms include: {', '.join(rare_terms)}."
                record1 = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                }
                dataset_records.append(record1)
                
                # Record 2: Rare terms if available
                if rare_terms:
                    prompt2 = f"List the rare terms associated with the document '{title}'."
                    response2 = (
                        f"The rare terms associated with the document '{title}' "
                        f"are: {', '.join(rare_terms)}."
                    )
                    record2 = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt2},
                            {"role": "assistant", "content": response2}
                        ]
                    }
                    dataset_records.append(record2)
                    
            elif mode == "raw_text":
                text_block = (
                    f"Taxonometry Profile for '{title}':\n"
                    f"- Zipf Average: {zipf_avg:.3f}\n"
                    f"- Rarity Position: {rarity_pos:.5f}"
                )
                if rare_terms:
                    text_block += f"\n- Rare Terms: {', '.join(rare_terms)}"
                dataset_records.append({"text": text_block})

    if not dataset_records:
        raise ValueError("No training records generated. Check inputs and mode settings.")
        
    # Shuffle and split
    random.seed(seed)
    random.shuffle(dataset_records)
    
    split_idx = int(len(dataset_records) * (1 - val_split))
    train_records = dataset_records[:split_idx]
    val_records = dataset_records[split_idx:]
    
    # Ensure val has at least some records if dataset is small
    if not val_records and len(dataset_records) > 1:
        train_records = dataset_records[:-1]
        val_records = dataset_records[-1:]
        
    # Write to output directory
    train_file = output_path / "train.jsonl"
    valid_file = output_path / "valid.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    with open(valid_file, "w", encoding="utf-8") as f:
        for rec in val_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    logger.info("Dataset creation complete:")
    logger.info(f"  Total records: {len(dataset_records)}")
    logger.info(f"  Training records saved to {train_file} ({len(train_records)} items)")
    logger.info(f"  Validation records saved to {valid_file} ({len(val_records)} items)")
    
    return len(train_records), len(val_records)
