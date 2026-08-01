"""Hardware manifest capture.

Every performance number in this project must be reported alongside the machine
and configuration that produced it - a latency figure without a hardware
manifest is not a measurement, it is an anecdote. Fields that cannot be
detected are reported as `None`/"unknown" rather than guessed.
"""

import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class HardwareManifest:
    platform: str
    machine: str
    processor: str
    python_version: str
    torch_version: str
    device: str
    cpu_count_logical: Optional[int]
    cpu_count_physical: Optional[int]
    total_ram_gb: Optional[float]
    torch_num_threads: int
    cuda_available: bool
    cuda_device_name: Optional[str] = None
    power_mode: str = "unknown"
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _power_mode() -> str:
    """Best-effort power/thermal mode. macOS exposes low-power mode via pmset;
    elsewhere we report "unknown" rather than assuming a state we can't read.
    This matters because a throttled or low-power machine produces latency
    numbers that are not comparable to a plugged-in one."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["pmset", "-g"], capture_output=True, text=True, timeout=5, check=True
            ).stdout
            for line in out.splitlines():
                if "lowpowermode" in line.replace(" ", "").lower():
                    value = line.strip().split()[-1]
                    return "low_power" if value == "1" else "normal"
        except Exception:  # noqa: BLE001
            return "unknown"
    return "unknown"


def capture_hardware_manifest(device: str = "cpu") -> HardwareManifest:
    cpu_logical: Optional[int] = None
    cpu_physical: Optional[int] = None
    ram_gb: Optional[float] = None
    try:
        import psutil

        cpu_logical = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:  # noqa: BLE001
        pass  # reported as None - never fabricated

    cuda_available = torch.cuda.is_available()
    return HardwareManifest(
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        device=device,
        cpu_count_logical=cpu_logical,
        cpu_count_physical=cpu_physical,
        total_ram_gb=ram_gb,
        torch_num_threads=torch.get_num_threads(),
        cuda_available=cuda_available,
        cuda_device_name=torch.cuda.get_device_name(0) if cuda_available else None,
        power_mode=_power_mode(),
    )
