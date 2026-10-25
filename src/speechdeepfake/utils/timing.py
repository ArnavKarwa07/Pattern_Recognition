"""Wall-clock timing helper."""

from __future__ import annotations

import time


class Timer:
    """Context manager measuring elapsed wall-clock seconds.

    >>> with Timer() as t:
    ...     pass
    >>> t.seconds >= 0
    True
    """

    def __init__(self) -> None:
        self.start = 0.0
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self.start
