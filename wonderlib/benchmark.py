import time
import logging
from typing import Any, Optional
import psutil
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class Benchmark(BaseModel):
    label: str = "Benchmark"
    input_tokens: int = -1
    output_tokens: int = 0
    token_throughput: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    cpu_start: Optional[Any] = Field(default=None, exclude=True)
    cpu_end: Optional[Any] = Field(default=None, exclude=True)
    proc: psutil.Process = Field(default_factory=psutil.Process, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    def start(self):
        self.start_time = time.perf_counter()
        self.cpu_start = self.proc.cpu_times()

    def stop(self):
        self.end_time = time.perf_counter()
        self.cpu_end = self.proc.cpu_times()
        elapsed = self.end_time - self.start_time
        if elapsed > 0:
            self.token_throughput = self.output_tokens / elapsed

    def report(self):
        elapsed = self.end_time - self.start_time
        cpu_time = (self.cpu_end.user + self.cpu_end.system) - (
            self.cpu_start.user + self.cpu_start.system
        )
        mem_info = self.proc.memory_info()

        logger.info(f"\n📊 {self.label}:")
        logger.info(f"- Elapsed wall time: {elapsed:.2f}s")
        logger.info(f"- CPU time (user+system): {cpu_time:.2f}s")
        logger.info(f"- Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")

        if self.output_tokens > 0:
            logger.info(f"- Token throughput: {self.token_throughput:.2f} tokens/sec")
            logger.info(
                f"- Input: {self.input_tokens} tokens, Output: {self.output_tokens} tokens"
            )
