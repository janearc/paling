import os
import sys
import logging
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

def main():
    repo_id = "mlx-community/Qwen2.5-14B-Instruct-4bit"
    files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    
    logger.info(f"Starting direct download of weights from {repo_id}...")
    for filename in files:
        try:
            logger.info(f"\nDownloading {filename}...")
            # hf_hub_download uses standard HTTPS and supports resume.
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                resume_download=True
            )
            logger.info(f"Successfully downloaded {filename} to {path}")
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            sys.exit(1)
            
    logger.info("\nAll weights downloaded successfully.")

if __name__ == "__main__":
    main()
