"""Rolling throughput measured from completed inference results."""

from collections import deque


class ResultRate:
    """Count completed intervals over roughly the last window of seconds.

    Keep the completion immediately before the window boundary so an interval
    crossing that boundary is measured in full, including slow detection frames.
    """

    def __init__(self, window_seconds=3.0):
        if window_seconds <= 0:
            raise ValueError("FPS window must be positive.")
        self.window_seconds = window_seconds
        self.completed = deque()
        self.context = None

    def update(self, now, context):
        if context != self.context:
            self.completed.clear()
            self.context = context
        if self.completed and now <= self.completed[-1]:
            self.completed.clear()
        self.completed.append(now)
        cutoff = now - self.window_seconds
        while len(self.completed) > 2 and self.completed[1] <= cutoff + 1e-9:
            self.completed.popleft()
        if len(self.completed) < 2:
            return None
        return (len(self.completed) - 1) / (now - self.completed[0])
