from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List, Any

import pandas as pd
import json
import torch

def load_file(load_path: Path) -> List[Dict] | pd.DataFrame | torch.Tensor:
    """
    Loads data from a JSON, CSV, or PyTorch file based on the file extension.

    Args:
        load_path: Path object pointing to the file to load
    Returns:
        Loaded data as a list of dicts (JSON), DataFrame (CSV), or Tensor (.pt)
    """
    if load_path.suffix == ".json":
        with open(load_path, "r") as f:
            return json.load(f)
    elif load_path.suffix == ".csv":
        return pd.read_csv(load_path)
    elif load_path.suffix == ".pt":
        return torch.load(load_path, weights_only=True)
    else:
        raise ValueError(f"Unsupported file type: {load_path}")

def ensure_dir(save_dir: Path) -> None:
    """
    Ensures the directory exists.
    """
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)


def save_file(data: Any, save_path: Path) -> None:
    """
    Saves data to a JSON, CSV, or PyTorch file based on the file extension.

    Args:
        data: Data to save (dict/list for JSON, DataFrame for CSV, Tensor for .pt)
        save_path: Path object pointing to the destination
    Returns:
        None
    """
    ensure_dir(save_path.parent)
    if save_path.suffix == ".json":
        with open(save_path, "w") as f:
            json.dump(data, f)
    elif save_path.suffix == ".csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(save_path, index=False)
        else:
            pd.DataFrame(data).to_csv(save_path, index=False)
    elif save_path.suffix == ".pt":
        torch.save(data, save_path)
    else:
        raise ValueError(f"Unsupported file type: {save_path}")
