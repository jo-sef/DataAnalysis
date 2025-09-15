"""Top-level package for data analysis helpers."""

from .config import RUNS_LOCATION, REPORT_FOLDER, SAMPLE_LIST, ALPHA1, ALPHA2
from .utils.merge import merge_measurements


def peakdet(*args, **kwargs):
    """Wrapper importing :func:`peakdet` on demand."""
    from .peakdet import peakdet as _peakdet
    return _peakdet(*args, **kwargs)

__all__ = [
    "RUNS_LOCATION",
    "REPORT_FOLDER",
    "SAMPLE_LIST",
    "ALPHA1",
    "ALPHA2",
    "peakdet",
    "merge_measurements",
]
