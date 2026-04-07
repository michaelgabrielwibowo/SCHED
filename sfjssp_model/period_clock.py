"""
PeriodClock — single source of truth for discrete period boundaries.

Maps a continuous time coordinate to a period index and computes
period start/end times, so that Worker (shift tracking) and
Operation (period_start/period_end bounds) use identical windows.
"""

from dataclasses import dataclass


@dataclass
class PeriodClock:
    """
    Converts a continuous time horizon into a sequence of fixed-length periods.

    Parameters
    ----------
    period_duration : float
        Length of each period in the same time unit as the schedule
        (default 480.0 = 8 hours in minutes, matching Worker.SHIFT_DURATION).
    horizon_start : float
        Time coordinate at which period 0 begins (default 0.0).
    """
    period_duration: float = 480.0
    horizon_start: float = 0.0

    def get_period(self, t: float) -> int:
        """Return the 0-indexed period that contains time t."""
        return int((t - self.horizon_start) // self.period_duration)

    def period_start(self, period_idx: int) -> float:
        """Return the start time of a given period."""
        return self.horizon_start + period_idx * self.period_duration

    def period_end(self, period_idx: int) -> float:
        """Return the end time (exclusive upper bound) of a given period."""
        return self.horizon_start + (period_idx + 1) * self.period_duration

    def periods_spanned(self, start: float, end: float) -> list[int]:
        """
        Return all period indices that overlap with [start, end).
        Useful for checking whether an operation crosses a period boundary.
        """
        p_start = self.get_period(start)
        p_end   = self.get_period(end - 1e-9)  # subtract epsilon for exclusive end
        return list(range(p_start, p_end + 1))

    def crosses_boundary(self, start: float, end: float) -> bool:
        """True if the interval [start, end) spans more than one period."""
        return self.get_period(start) != self.get_period(end - 1e-9)
