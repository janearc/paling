import json
import os
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# A default system prompt for knowledge injection
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant with access to the user's personal notes and documentation. Use this knowledge to answer questions accurately and concisely."

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
                headers_path = [current_headers[i] for i in range(1, current_level + 1) if current_headers[i]]
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
        headers_path = [current_headers[i] for i in range(1, current_level + 1) if current_headers[i]]
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

def chunk_text_by_tokens(text: str, tokenizer: Any, chunk_size: int = 1024, overlap: int = 128) -> List[str]:
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
        print(f"Warning: RLHF directory '{rlhf_dir}' does not exist. Skipping.")
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
            print(f"Warning: Failed to parse RLHF file '{file}': {e}")
            
    print(f"Parsed {len(rlhf_data)} approved QA pairs from RLHF reviews.")
    return rlhf_data

def parse_taxonometry_directory(tax_dir: Path) -> List[Dict[str, Any]]:
    """
    Parses taxonometry metrics from a directory.
    """
    tax_data = []
    if not tax_dir.exists():
        print(f"Warning: Taxonometry directory '{tax_dir}' does not exist. Skipping.")
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
            print(f"Warning: Failed to parse Taxonometry file '{file}': {e}")
            
    print(f"Parsed {len(tax_data)} Taxonometry profile definitions.")
    return tax_data

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
    """
    Processes markdown files, RLHF data, and taxonometry profiles to build train/validation JSONL datasets.
    """
    if exclude_patterns is None:
        exclude_patterns = [r'\.git', r'\.venv', r'\.obsidian', r'node_modules']
        
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    md_files = extract_markdown_files(input_path, exclude_patterns)
    if not md_files:
        raise ValueError(f"No markdown files found in {input_dir}")
        
    print(f"Found {len(md_files)} markdown files.")
    
    # Try loading tokenizer if model_path is provided and mode requires chunking
    tokenizer = None
    if model_path and mode == "raw_text":
        try:
            print(f"Loading tokenizer from {model_path} for token-based chunking...")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load tokenizer ({e}). Falling back to word-based chunking.")
            
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
                    prompt = f"In the document '{sec['file_name']}' under '{path_str}', what is written?"
                    
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
                    grounded_text = f"Document: {rel_path}\nChunk {idx+1}/{len(chunks)}\n---\n{chunk}"
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
            print(f"Error processing {md_file}: {e}")
            
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
                prompt = f"What is the taxonometric profile of the sigil '{title}'?"
                response = f"The sigil '{title}' has a Zipf average of {zipf_avg:.3f} and rarity position of {rarity_pos:.5f}."
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
                    prompt2 = f"List the rare terms associated with the sigil '{title}'."
                    response2 = f"The rare terms associated with the sigil '{title}' are: {', '.join(rare_terms)}."
                    record2 = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt2},
                            {"role": "assistant", "content": response2}
                        ]
                    }
                    dataset_records.append(record2)
                    
            elif mode == "raw_text":
                text_block = f"Taxonometry Profile for '{title}':\n- Zipf Average: {zipf_avg:.3f}\n- Rarity Position: {rarity_pos:.5f}"
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
            
    print(f"Dataset creation complete:")
    print(f"  Total records: {len(dataset_records)}")
    print(f"  Training records saved to {train_file} ({len(train_records)} items)")
    print(f"  Validation records saved to {valid_file} ({len(val_records)} items)")
    
    return len(train_records), len(val_records)
