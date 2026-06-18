import os
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("paling-munge")

def main():
    bento_dir = os.environ.get("PALING_BENTOS_ROOT", str(Path.home() / "var" / "paling" / "bentos"))
    logger.info("paling-munge starting up")
    logger.info(f"monitoring drop-zone: {bento_dir}")
    
    if not os.path.exists(bento_dir):
        logger.error(f"Drop-zone {bento_dir} does not exist. Ensure volume is mounted correctly.")
        return

    # Basic polling loop for demonstration before we add watchdog
    seen = set(os.listdir(bento_dir))
    
    try:
        while True:
            current = set(os.listdir(bento_dir))
            new_bentos = current - seen
            
            for bento in new_bentos:
                bento_path = os.path.join(bento_dir, bento)
                if os.path.isdir(bento_path):
                    logger.info(f"new bento detected: {bento}")
                    # TODO: Trigger preflight checks and adapter logic here
                    
            seen = current
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("paling-munge shutting down.")

if __name__ == "__main__":
    main()
