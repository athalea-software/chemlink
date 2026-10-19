"""Hardware summary for the TUI header, detected once in a background thread so
the first frame never waits on ``nvidia-smi``."""
from __future__ import annotations

import threading
from typing import Optional

from rich.text import Text


class SystemInfo:
    def __init__(self) -> None:
        self.ready = False
        self.gpu: Optional[str] = None
        self.gpu_count = 0
        self.cpu = ""
        self.ram = ""
        self._thread = threading.Thread(target=self._detect, daemon=True)
        self._thread.start()

    def _detect(self) -> None:
        try:
            try:
                from ...hpc.cluster.resource_detector import get_hardware_profile
            except ImportError:
                from hpc.cluster.resource_detector import get_hardware_profile  # type: ignore
            hw = get_hardware_profile()
            if hw.has_gpu:
                first = hw.gpus[0]
                name = first.name.replace("NVIDIA ", "").replace("GeForce ", "")
                self.gpu = f"{name} {first.memory_total_gb:.0f} GB"
                self.gpu_count = hw.gpu_count
            self.cpu = f"{hw.cpu_threads or hw.cpu_cores} threads" if (hw.cpu_threads or hw.cpu_cores) else ""
            self.ram = f"{hw.ram_total_gb:.0f} GB RAM" if hw.ram_total_gb else ""
        except Exception:
            pass
        finally:
            self.ready = True

    def chips(self) -> Text:
        out = Text(no_wrap=True)
        if not self.ready:
            out.append("detecting hardware…", style="cl.faint")
            return out
        if self.gpu:
            out.append("● ", style="cl.ok.soft")
            label = f"{self.gpu_count}× {self.gpu}" if self.gpu_count > 1 else self.gpu
            out.append(label, style="cl.muted")
        else:
            out.append("○ ", style="cl.warn.soft")
            out.append("no GPU · CPU mode", style="cl.muted")
        for part in (self.cpu, self.ram):
            if part:
                out.append("  ·  ", style="cl.faint")
                out.append(part, style="cl.muted")
        return out
