"""Compatibility re-export for adaptive window helpers."""
from src.models.dit.window import WindowPlan, WindowSlices, compute_adaptive_windows

__all__ = ["WindowPlan", "WindowSlices", "compute_adaptive_windows"]
