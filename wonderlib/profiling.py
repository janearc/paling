import re
import json
import logging
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Any, Callable, Tuple

import spacy
import torch
from pydantic import BaseModel, Field, ConfigDict

from wonderlib.benchmark import Benchmark
from wonderlib.markdown_xml import markdown_to_xml
from wordfreq import zipf_frequency
from wonderlib.git_stats import GitStats, get_git_stats

# Setup local logger
logger = logging.getLogger("wonderlib")

# Try to load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    nlp = None
    logger.info(f"⚠️ spaCy en_core_web_sm failed to load: {e}")

class TaxonometryProfile(BaseModel):
    title: str
    zipf_avg: float
    zipf_cluster: List[int]
    rare_terms: List[str]
    rarity_pos: float  # 0-1 normalized rarity via POS tags

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    benchmark: Optional[Benchmark] = None
    filename: Optional[str] = Field(
        None, description="The source filename of the signature (for tracking)."
    )
    git_stats: Optional[GitStats] = Field(
        None, description="git statistics for the document being profiled."
    )

    @property
    def rare_term_count(self) -> int:
        return len(self.rare_terms)

    @property
    def zipf_high(self) -> int:
        return self.zipf_cluster[0]

    @property
    def zipf_med(self) -> int:
        return self.zipf_cluster[1]

    @property
    def zipf_low(self) -> int:
        return self.zipf_cluster[2]

class TaxonometryCorpus(BaseModel):
    signatures: List[TaxonometryProfile] = Field(
        ..., description="a full set of document taxonometric signatures"
    )

    @property
    def length(self) -> int:
        return len(self.signatures)

    def pop(self) -> TaxonometryProfile:
        return self.signatures.pop(0)

    def push(self, signature: TaxonometryProfile):
        self.signatures.append(signature)

    def no_rare_terms(self) -> List[TaxonometryProfile]:
        return [sig for sig in self.signatures if sig.rare_term_count == 0]

    def no_rare_term_filenames(self) -> List[str]:
        return [
            sig.filename for sig in self.signatures
            if sig.rare_term_count == 0 and sig.filename is not None
        ]

class RarityAnalyzer:
    def __init__(
        self,
        token_count: int,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.benchmark = Benchmark(label="lexical_rarity", input_tokens=token_count)
        self.benchmark.start()

        self.model = model
        self.tokenizer = tokenizer

    def get_zipf_score(self, text: str, lang: str = "en") -> float:
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.0
        scores = [zipf_frequency(word, lang, wordlist="large") for word in words]
        return sum(scores) / len(scores)

    def get_zipf_cluster(self, text: str, lang: str = "en") -> List[int]:
        cluster = [0, 0, 0]  # [0–3] high, [3–5] medium, [5+] low
        words = re.findall(r"\b\w+\b", text.lower())
        for word in words:
            score = zipf_frequency(word, lang, wordlist="large")
            if score < 3:
                cluster[0] += 1
            elif score < 5:
                cluster[1] += 1
            else:
                cluster[2] += 1
        return cluster

    def get_pos_rarity(self, text: str) -> float:
        if not nlp:
            return 0.5  # fallback neutral score
        doc = nlp(text)
        rare_tags = {"X", "SYM", "NUM", "INTJ", "PROPN"}
        rare_count = sum(1 for token in doc if token.pos_ in rare_tags)
        total = len(doc)
        if total == 0:
            return 0.5
        return rare_count / total

    def extract_rare_terms_heuristically(self, context: str) -> List[str]:
        """
        Extracts rare words using purely offline lexical/statistical rules (Zipf frequency + POS tags).
        Requires zero model execution and runs in milliseconds.
        """
        seen = set()
        rare_terms = []
        
        if not nlp:
            # Fallback if spaCy is unavailable
            words = re.findall(r"\b[a-zA-Z-]{4,}\b", context)
            for word in words:
                word_lower = word.lower()
                if word_lower not in seen:
                    seen.add(word_lower)
                    # Low-frequency Zipf words
                    if zipf_frequency(word_lower, "en", wordlist="large") < 2.5:
                        rare_terms.append(word)
            return rare_terms[:15]
            
        doc = nlp(context)
        for token in doc:
            word = token.text.strip()
            # Select proper nouns, nouns, and adjectives of length >= 3
            if len(word) >= 3 and token.pos_ in {"PROPN", "NOUN", "ADJ"} and word.isalpha():
                word_lower = word.lower()
                if word_lower not in seen:
                    seen.add(word_lower)
                    # Zipf threshold < 3.0 represents rare/uncommon words in English
                    if zipf_frequency(word_lower, "en", wordlist="large") < 3.0:
                        rare_terms.append(word)
                        
        return rare_terms[:15]

    def extract_rare_terms(self, context: str, local_logger: Optional[Any] = None) -> List[str]:
        log = local_logger or logger
        
        # If no model is provided, default to the fast, local lexical heuristic
        if not self.model or not self.tokenizer:
            log.debug("No model provided for rare term extraction. Using lexical POS/Zipf heuristics.")
            terms = self.extract_rare_terms_heuristically(context)
            joined = ", ".join(terms)
            self.benchmark.output_tokens = len(joined.split()) * 4 // 3
            return terms

        prompt = (
            "Given the passage below, starting with <|PASSAGE|> and ending with <|END|>. "
            "Identify and extract unusual words or phrases, comprising from one to three words. "
            "Respond with a list of these terms, each starting with the marker ##.\n\n"
            f"<|PASSAGE|>\n\n{context}\n\n<|END|>"
        )

        # 1. MLX-specific generation
        is_mlx_model = hasattr(self.model, "layers") or "mlx" in str(type(self.model)).lower()
        if is_mlx_model:
            try:
                log.debug("Running MLX-native generation for rare terms extraction...")
                from mlx_lm import generate
                result = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=128,
                    temp=0.7
                )
            except Exception as e:
                log.error(f"Failed MLX generation: {e}. Falling back to lexical heuristics.")
                return self.extract_rare_terms_heuristically(context)
        # 2. PyTorch Transformers generation
        else:
            try:
                log.debug("Running PyTorch Transformers generation for rare terms extraction...")
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs, max_new_tokens=128, temperature=0.7, top_k=750, top_p=0.95
                    )
                result = self.tokenizer.decode(output[0], skip_special_tokens=True)
            except Exception as e:
                log.error(f"Failed PyTorch generation: {e}. Falling back to lexical heuristics.")
                return self.extract_rare_terms_heuristically(context)

        log.debug(f"🔍 Full LLM response: {result}")

        terms = []
        seen = set()
        for line in result.splitlines():
            match = re.search(r"##\s*(.+)", line)
            if match:
                term = match.group(1).strip()
                if term and term[-1].isalpha() and term not in seen:
                    seen.add(term)
                    terms.append(term)

        log.debug(f"✅ Cleaned rare terms: {terms}")
        
        joined = ", ".join(terms)
        self.benchmark.output_tokens = len(joined.split()) * 4 // 3

        return terms

def profile_document(
    text: str,
    title: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    token_estimator: Optional[Callable[[str], int]] = None,
    local_logger: Optional[Any] = None
) -> TaxonometryProfile:
    """
    Profiles a markdown string and returns a TaxonometryProfile taxonometry object.
    """
    log = local_logger or logger
    
    # Unparse markdown structure via XML parsing
    root = markdown_to_xml(text)
    paragraphs = [elem.text for elem in root.findall("p") if elem.text]
    context = "\n\n".join(paragraphs)

    # Estimate inputs tokens
    if token_estimator:
        token_count = token_estimator(context)
    else:
        token_count = len(context.split()) * 4 // 3

    # Instantiate analyzer
    analyzer = RarityAnalyzer(
        token_count=token_count,
        model=model,
        tokenizer=tokenizer
    )

    # Run analysis
    zipf_avg = analyzer.get_zipf_score(context)
    zipf_cluster = analyzer.get_zipf_cluster(context)
    rarity_pos = analyzer.get_pos_rarity(context)
    rare_terms = analyzer.extract_rare_terms(context, log)

    analyzer.benchmark.stop()

    return TaxonometryProfile(
        title=title or "unknown",
        zipf_avg=zipf_avg,
        zipf_cluster=zipf_cluster,
        rarity_pos=rarity_pos,
        rare_terms=rare_terms,
        benchmark=analyzer.benchmark,
    )

def DataToTaxonometryCorpus(data: str) -> TaxonometryCorpus:
    root = Path(data)
    files = root.glob("**/*-taxonometry.json")
    signatures = []

    for file in files:
        try:
            with open(file, "r") as f:
                raw_json = json.load(f)
                signature = TaxonometryProfile(**raw_json)
                signature.filename = str(file)
                signatures.append(signature)
        except Exception as e:
            raise RuntimeError(f"Failed to load {file}: {e}")

    return TaxonometryCorpus(signatures=signatures)
