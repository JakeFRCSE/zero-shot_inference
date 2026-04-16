from __future__ import annotations

import hashlib
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

def resolve_save_dir(
    model_name: str,
    dataset_name: str,
    prompt_template: str,
    results_dir: Path = Path('results'),
) -> Path:
    safe_model_name = model_name.replace('/', '_')
    prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:8]
    return results_dir / safe_model_name / dataset_name / f'prompt_{prompt_hash}'


def load_experiment_results(
    model_name: str,
    dataset_name: str,
    prompt_template: str,
    results_dir: Path = Path('results'),
) -> Dict[str, Any]:
    """
    Loads all saved experiment artifacts (evaluation CSVs and .pt tensors)
    from the directory determined by model_name, dataset_name, and prompt_template.

    Args:
        model_name: HuggingFace model identifier (e.g. 'meta-llama/Llama-3.2-3B')
        dataset_name: Dataset stem name (e.g. 'antonym')
        prompt_template: Prompt format string used during the experiment
        results_dir: Root results directory

    Returns:
        Dict with keys:
            'save_dir'          – Path to the resolved directory
            'eval_dfs'          – Dict[str, DataFrame] mapping relation name → eval CSV
            'mean_vector'       – Tensor (if exists)
            'score'             – Tensor (if exists)
            'relation_vector'   – Tensor (if exists)
    """
    save_dir = resolve_save_dir(model_name, dataset_name, prompt_template, results_dir)
    if not save_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {save_dir}")

    results: Dict[str, Any] = {'save_dir': save_dir}

    eval_dfs: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(save_dir.glob('_*.csv')):
        relation = csv_path.stem[1:]
        eval_dfs[relation] = pd.read_csv(csv_path)
    results['eval_dfs'] = eval_dfs

    for pt_name in ['mean_vector', 'score', 'relation_vector']:
        pt_path = save_dir / f'{pt_name}.pt'
        if pt_path.exists():
            results[pt_name] = torch.load(pt_path, weights_only=True)

    return results


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
