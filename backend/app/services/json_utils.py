"""Utilities for JSON serialization."""
import numpy as np

def make_json_safe(obj):
    """Recursively convert numpy types to JSON-safe Python types."""
    if hasattr(obj, 'item'):
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return 0.0
        return val
    elif hasattr(obj, 'tolist'):
        return make_json_safe(obj.tolist())
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, np.floating):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    elif isinstance(obj, np.integer):
        return int(obj)
    return obj
