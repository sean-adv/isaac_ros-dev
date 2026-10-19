"""Reject unreliable depth components before plane fitting."""

import numpy as np


class CloudFilter:
    """Configuration is shared; mutable history belongs to one worker object ID."""

    def __init__(self, min_coverage=.15, min_coverage_ratio=.35,
                 max_depth_jump=.10, confirmations=3):
        for name, value in (("min_coverage", min_coverage),
                            ("min_coverage_ratio", min_coverage_ratio)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if not np.isfinite(max_depth_jump) or max_depth_jump <= 0:
            raise ValueError("max_depth_jump must be finite and positive.")
        if type(confirmations) is not int or confirmations < 1:
            raise ValueError("confirmations must be an integer >= 1.")
        self.min_coverage = min_coverage
        self.min_coverage_ratio = min_coverage_ratio
        self.max_depth_jump = max_depth_jump
        self.confirmations = confirmations

    def check(self, coverage, depth, state, frame=None):
        """Reject sparse fragments permanently; confirm coherent regime changes.

        Coverage is the retained sample count / clipped bbox pixel area. It is
        normalized for 2D box size changes. Only distinct depth stamps count.
        """
        if frame is not None and state.get("last_frame") == frame:
            if state.get("last_error"):
                raise ValueError(state["last_error"])
            return
        state["last_frame"] = frame
        state["last_error"] = None
        if coverage < self.min_coverage:
            state.pop("pending", None)
            state["last_error"] = "insufficient_cloud_coverage"
            raise ValueError(state["last_error"])
        previous = state.get("accepted")
        changed = previous is not None and (
            coverage < previous[0] * self.min_coverage_ratio
            or abs(depth - previous[1]) > self.max_depth_jump)
        if changed:
            pending = state.get("pending")
            consistent = pending is not None and (
                abs(depth - pending[1]) <= self.max_depth_jump / 2
                and min(coverage, pending[0]) >= max(coverage, pending[0]) * .5)
            count = pending[2] + 1 if consistent else 1
            state["pending"] = (coverage, depth, count)
            if count < self.confirmations:
                state["last_error"] = f"cloud_change_pending:{count}/{self.confirmations}"
                raise ValueError(state["last_error"])
        state["accepted"] = (coverage, depth)
        state.pop("pending", None)
