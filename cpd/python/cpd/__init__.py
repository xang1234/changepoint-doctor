"""Python bindings for cpd-rs."""

from ._cpd_rs import (
    Binseg,
    Diagnostics,
    OfflineChangePointResult,
    Pelt,
    PruningStats,
    SegmentStats,
    SmokeDetector,
    __version__,
    detect_offline,
    smoke_detect,
)

__all__ = [
    "__version__",
    "PruningStats",
    "SegmentStats",
    "Diagnostics",
    "OfflineChangePointResult",
    "Pelt",
    "Binseg",
    "detect_offline",
    "SmokeDetector",
    "smoke_detect",
]
