import sys
import traceback
from pathlib import Path
from typing import List, Dict, Optional
from mlx_lm import load, stream_generate

class ChatSession:
    def __init__(self, model_path: str, adapter_path: Optional[str] = None, system_prompt: Optional[str] = None):
        print(f"Loading model from '{model_path}'...")
        
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
                print("⚠️ Warning: LoRA adapters are not supported for GGUF models in this mode. Running base model only.")
                
            try:
                import llama_cpp
            except ImportError:
                print("\n" + "=" * 80)
                print("❌ Error: To run GGUF models, you must install 'llama-cpp-python' in your environment.")
                print("To compile with hardware Metal GPU Acceleration on Apple Silicon, run:")
                print("  CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python")
                print("Or using poetry:")
                print("  CMAKE_ARGS='-DLLAMA_METAL=on' poetry run pip install llama-cpp-python")
                print("=" * 80 + "\n")
                sys.exit(1)
                
            print(f"Initializing GGUF model from '{self.gguf_path}' via llama_cpp (Metal GPU enabled)...")
            try:
                self.model = llama_cpp.Llama(
                    model_path=str(self.gguf_path),
                    n_ctx=2048,
                    n_gpu_layers=-1, # Map all layers to GPU via Metal
                    verbose=False
                )
            except Exception as e:
                print(f"❌ Failed to load GGUF model: {e}")
                raise e
            self.tokenizer = None
        
        # 3. Initialize MLX model via mlx-lm
        else:
            if adapter_path:
                print(f"Loading LoRA adapter from '{adapter_path}'...")
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
                        print(delta['content'], end='', flush=True)
                        response_text += delta['content']
                print() # Print final newline
            except KeyboardInterrupt:
                print("\n[Generation interrupted]")
            except Exception as e:
                print(f"\n[Error during GGUF generation: {e}]")
                traceback.print_exc()
        
        # 2. MLX Streaming Inference
        else:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    self.history,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
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
                    print(response.text, end="", flush=True)
                    response_text += response.text
                print() # Print final newline
            except KeyboardInterrupt:
                print("\n[Generation interrupted]")
            except Exception as e:
                print(f"\n[Error during MLX generation: {e}]")
                traceback.print_exc()
                
        if response_text:
            self.history.append({"role": "assistant", "content": response_text})
        return response_text

def run_interactive_chat(model_path: str, adapter_path: Optional[str] = None, system_prompt: Optional[str] = None):
    """
    Launches an interactive shell chat session.
    """
    try:
        session = ChatSession(model_path, adapter_path, system_prompt)
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return

    print("\n" + "=" * 50)
    print("Interactive Chat Session Started.")
    print("Commands:")
    print("  /exit or /quit - Exit the session")
    print("  /clear         - Clear conversation history")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You > ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        cleaned_input = user_input.strip()
        if not cleaned_input:
            continue

        if cleaned_input.lower() in ["/exit", "/quit"]:
            print("Exiting.")
            break

        if cleaned_input.lower() == "/clear":
            session.reset_history()
            print("Conversation history cleared.")
            continue

        print("Assistant > ", end="", flush=True)
        session.send_message(cleaned_input)
        print()
