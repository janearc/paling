# ==============================================================================
# Paling Inference Engine (`paling chat`)
# ==============================================================================
# This module provides a unified interactive chat session layer, abstracting away
# the complexity of running diverse ML weights on Apple Silicon (Metal API).
#
# Core Architectural Features:
# 1. Dual-Backend Execution: Dynamically routes inference to either:
#    - `mlx_lm` (Native MLX): Used for natively quantized MLX weights and 
#      runtime LoRA adapter injection.
#    - `llama-cpp-python` (GGUF): Used for executing pre-fused GGUF binaries 
#      with hardware Metal acceleration (requires `-DLLAMA_METAL=on`).
# 2. Dynamic Format Detection: Automatically inspects the provided model path
#    to detect `.gguf` binaries versus standard HuggingFace/MLX snapshot directories.
# 3. Chat History & Templating: Manages conversation context state across turns
#    and strictly enforces tokenizer chat templates for safety and alignment.
# 4. Token Streaming: Preserves terminal UX by streaming tokens in real-time
#    using direct `sys.stdout.write` instead of fully buffering generation.
# ==============================================================================
import sys
import logging
import traceback
from pathlib import Path
from typing import Optional
from mlx_lm import load, stream_generate

logger = logging.getLogger(__name__)

class ChatSession:
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        logger.info(f"Loading model from '{model_path}'...")
        
        self.system_prompt = system_prompt
        self.history = []
        
        # 1. Detect GGUF model format
        self.is_gguf = False
        self.gguf_path = None
        
        p = Path(model_path)
        if p.is_file() and p.suffix == ".gguf":
            self.is_gguf = True
            self.gguf_path = p
        elif p.is_dir():
            # Search for a .gguf file recursively (e.g. inside snapshot folder)
            ggufs = list(p.glob("**/*.gguf"))
            if ggufs:
                self.is_gguf = True
                self.gguf_path = ggufs[0]
                
        # 2. Initialize GGUF model via llama-cpp-python
        if self.is_gguf:
            if adapter_path:
                logger.info(
                    "Warning: LoRA adapters are not supported for GGUF models in "
                    "this mode. Running base model only."
                )
                
            try:
                import llama_cpp
            except ImportError:
                logger.info("\n" + "=" * 80)
                logger.info(
                    "Error: To run GGUF models, you must install "
                    "'llama-cpp-python' in your environment."
                )
                logger.info(
                    "To compile with hardware Metal GPU Acceleration on Apple Silicon, run:"
                )
                logger.info("  CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python")
                logger.info("Or using poetry:")
                logger.info(
                    "  CMAKE_ARGS='-DLLAMA_METAL=on' poetry run pip install llama-cpp-python"
                )
                logger.info("=" * 80 + "\n")
                sys.exit(1)
                
            logger.info(
                f"Initializing GGUF model from '{self.gguf_path}' "
                "via llama_cpp (Metal GPU enabled)..."
            )
            try:
                self.model = llama_cpp.Llama(
                    model_path=str(self.gguf_path),
                    n_ctx=2048,
                    n_gpu_layers=-1, # Map all layers to GPU via Metal
                    verbose=False
                )
            except Exception as e:
                logger.info(f"❌ Failed to load GGUF model: {e}")
                raise e
            self.tokenizer = None
        
        # 3. Initialize MLX model via mlx-lm
        else:
            if adapter_path:
                logger.info(f"Loading LoRA adapter from '{adapter_path}'...")
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
            
        self.reset_history()

    def reset_history(self):
        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

    def send_message(self, user_message: str, max_tokens: int = 1024, temp: float = 0.7) -> str:
        self.history.append({"role": "user", "content": user_message})
        
        response_text = ""
        
        # 1. GGUF Streaming Inference
        if self.is_gguf:
            try:
                response = self.model.create_chat_completion(
                    messages=self.history,
                    stream=True,
                    max_tokens=max_tokens,
                    temperature=temp
                )
                for chunk in response:
                    delta = chunk['choices'][0]['delta']
                    if 'content' in delta:
                        sys.stdout.write(str(delta['content']))
                        sys.stdout.flush()
                        response_text += delta['content']
                logger.info("") # Print final newline
            except KeyboardInterrupt:
                logger.info("\n[Generation interrupted]")
            except Exception as e:
                logger.info(f"\n[Error during GGUF generation: {e}]")
                traceback.print_exc()
        
        # 2. MLX Streaming Inference
        else:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    self.history,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback for models without standard chat templates
                prompt = ""
                for msg in self.history:
                    if msg["role"] == "system":
                        prompt += f"System: {msg['content']}\n"
                    elif msg["role"] == "user":
                        prompt += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        prompt += f"Assistant: {msg['content']}\n"
                prompt += "Assistant: "

            try:
                for response in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=temp
                ):
                    sys.stdout.write(str(response.text))
                    sys.stdout.flush()
                    response_text += response.text
                logger.info("") # Print final newline
            except KeyboardInterrupt:
                logger.info("\n[Generation interrupted]")
            except Exception as e:
                logger.info(f"\n[Error during MLX generation: {e}]")
                traceback.print_exc()
                
        if response_text:
            self.history.append({"role": "assistant", "content": response_text})
        return response_text

def run_interactive_chat(
    model_path: str,
    adapter_path: Optional[str] = None,
    system_prompt: Optional[str] = None,
):
    """Run an interactive shell chat session against the model (optionally with an adapter)."""
    # build the session up front so a bad model/adapter path fails immediately
    # with a clear message, rather than after the prompt loop has started.
    try:
        session = ChatSession(model_path, adapter_path, system_prompt)
    except Exception as e:
        logger.info(f"Error loading model: {e}")
        traceback.print_exc()
        return

    logger.info("\n" + "=" * 50)
    logger.info("Interactive Chat Session Started.")
    logger.info("Commands:")
    logger.info("  /exit or /quit - Exit the session")
    logger.info("  /clear         - Clear conversation history")
    logger.info("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You > ")
        except (KeyboardInterrupt, EOFError):
            logger.info("\nExiting.")
            break

        cleaned_input = user_input.strip()
        if not cleaned_input:
            continue

        if cleaned_input.lower() in ["/exit", "/quit"]:
            logger.info("Exiting.")
            break

        if cleaned_input.lower() == "/clear":
            session.reset_history()
            logger.info("Conversation history cleared.")
            continue

        sys.stdout.write(str("Assistant > "))
        sys.stdout.flush()
        session.send_message(cleaned_input)
        logger.info("")
