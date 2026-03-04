from __future__ import annotations

import ast
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None


def _to_numpy_array(value: Any):
    if isinstance(value, np.ndarray):
        return value
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return None


def deep_equal(expected: Any, actual: Any) -> bool:
    expected_array = _to_numpy_array(expected)
    actual_array = _to_numpy_array(actual)
    if expected_array is not None or actual_array is not None:
        if expected_array is None:
            expected_array = np.asarray(expected)
        if actual_array is None:
            actual_array = np.asarray(actual)
        try:
            if np.array_equal(expected_array, actual_array):
                return True
        except Exception:
            pass
        try:
            return bool(np.allclose(expected_array, actual_array, equal_nan=True))
        except Exception:
            return False
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(deep_equal(e, a) for e, a in zip(expected, actual))
    if isinstance(expected, set) and isinstance(actual, set):
        if len(expected) != len(actual):
            return False
        unmatched = list(actual)
        for item in expected:
            for idx, candidate in enumerate(list(unmatched)):
                if deep_equal(item, candidate):
                    unmatched.pop(idx)
                    break
            else:
                return False
        return not unmatched
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return False
        return all(deep_equal(expected[key], actual[key]) for key in expected)
    return expected == actual


def to_bool(value: Any) -> bool:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False
            return False
    array_value = _to_numpy_array(value)
    if array_value is not None:
        try:
            return bool(np.all(array_value))
        except Exception:
            try:
                return bool(np.all(array_value != 0))
            except Exception:
                return bool(np.any(array_value))
    if hasattr(value, "all"):
        try:
            return bool(value.all())
        except Exception:
            return bool(value)
    if isinstance(value, (list, tuple, set)):
        return all(to_bool(item) for item in value)
    if isinstance(value, dict):
        return all(to_bool(v) for v in value.values())
    return bool(value)
