import time
import psutil
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TelemetryReport:
    # Memory (in MB)
    peak_cpu_memory_mb: float = 0.0
    peak_gpu_memory_mb: Optional[float] = None
    
    # Timing (in ms)
    encode_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    prefill_latency_ms: float = 0.0
    
    # Throughput (tokens/s)
    decode_throughput_tps: float = 0.0
    
    # Is it estimated or measured?
    measured: bool = True

class TelemetryTracker:
    """Tracks peak memory and execution times for codecs and models."""
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.process = psutil.Process()
        self.reports: List[TelemetryReport] = []
        
        self._start_cpu_mem = 0.0
        self._start_gpu_mem = 0.0
        self._start_time = 0.0

    def get_cpu_memory_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def get_gpu_memory_mb(self) -> float:
        if self.device != "cpu" and torch.cuda.is_available():
            # Returns max memory allocated so far
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0
        
    def reset_gpu_peaks(self):
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def start_recording(self):
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.reset_gpu_peaks()
        self._start_cpu_mem = self.get_cpu_memory_mb()
        self._start_gpu_mem = self.get_gpu_memory_mb()
        self._start_time = time.perf_counter()

    def end_recording(self, context: str) -> Dict[str, Optional[float]]:
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        peak_cpu = self.get_cpu_memory_mb() - self._start_cpu_mem
        
        # We capture the max memory recorded during the interval
        peak_gpu = None
        if self.device != "cpu" and torch.cuda.is_available():
            peak_gpu = self.get_gpu_memory_mb() - self._start_gpu_mem
            
        return {
            "elapsed_ms": elapsed_ms,
            "peak_cpu_mb": peak_cpu,
            "peak_gpu_mb": peak_gpu
        }

    def measure_throughput(self, func, num_tokens: int, warmup_steps: int = 5, measure_steps: int = 10) -> float:
        """Measure tokens per second by executing a function repeatedly."""
        # Warm-up
        for _ in range(warmup_steps):
            func()
            
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start = time.perf_counter()
        for _ in range(measure_steps):
            func()
            
        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        total_time = time.perf_counter() - start
        
        # total_time is for measure_steps invocations, each generating `num_tokens` (usually 1 for decode)
        return (num_tokens * measure_steps) / total_time
