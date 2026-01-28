import os
import time
import threading
import psutil

class CPUMonitor:
    """
    Samples per-process CPU percent periodically.
    Note: cpu_percent can exceed 100% when multiple cores are used (e.g. 400%).
    """
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None
        self.samples = []

    def _run(self):
        # Prime the measurement (first call is always 0.0-ish)
        self._proc.cpu_percent(interval=None)
        while not self._stop.is_set():
            self.samples.append(self._proc.cpu_percent(interval=None))
            time.sleep(self.interval_s)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def avg_cpu_percent(self) -> float:
        return (sum(self.samples) / len(self.samples)) if self.samples else 0.0

    @property
    def peak_cpu_percent(self) -> float:
        return max(self.samples) if self.samples else 0.0
