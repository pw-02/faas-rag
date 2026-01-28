import threading
import psutil
import os
import time


class MemoryMonitor:
    """
    Samples process RSS periodically in a background thread.
    Tracks average and peak RSS over the monitoring window.
    """
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = None

        self.samples = 0
        self.sum_rss = 0
        self.peak_rss = 0

    def _run(self):
        # warm first read
        try:
            rss = self._proc.memory_info().rss
        except Exception:
            rss = 0
        self.peak_rss = max(self.peak_rss, rss)

        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss  # bytes
            except Exception:
                rss = 0
            self.samples += 1
            self.sum_rss += rss
            if rss > self.peak_rss:
                self.peak_rss = rss
            time.sleep(self.interval_s)

    def start(self):
        self._stop.clear()
        self.samples = 0
        self.sum_rss = 0
        self.peak_rss = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def avg_rss_bytes(self) -> int:
        return int(self.sum_rss / self.samples) if self.samples else 0

def bytes_to_mib(b: int) -> float:
    return b / (1024 * 1024)

def bytes_to_gb(b: int) -> float:
    return b / (1024 * 1024 * 1024)
