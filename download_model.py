import os
import sys
from huggingface_hub import hf_hub_download

def main():
    repo_id = "mlx-community/Qwen2.5-14B-Instruct-4bit"
    files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    
    print(f"Starting direct download of weights from {repo_id}...")
    for filename in files:
        try:
            print(f"\nDownloading {filename}...")
            # hf_hub_download uses standard HTTPS and supports resume.
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                resume_download=True
            )
            print(f"Successfully downloaded {filename} to {path}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}", file=sys.stderr)
            sys.exit(1)
            
    print("\nAll weights downloaded successfully.")

if __name__ == "__main__":
    main()
