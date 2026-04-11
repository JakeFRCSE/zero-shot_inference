from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across random, torch, and cuda.

    Args:
        seed: The seed value to use
    Returns:
        None
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_correct_samples(df: pd.DataFrame, correct: bool = True) -> pd.DataFrame:
    """
    Filters the DataFrame based on whether the model predicted the correct output.

    Args:
        df: Input DataFrame with evaluation results
        correct: If True, keep samples where output_prediction is correct.
                 If False, keep samples where output_prediction is incorrect.
    Returns:
        Filtered DataFrame with selected columns
    """
    columns = ["input", "output", "relation", "prompt", "input_id", "output_id", "relation_id"]

    filtered_df = df[df["output_prediction"] == correct].copy()

    filtered_df = filtered_df[columns]

    filtered_df = filtered_df.reset_index(drop=True)
    label = "correct" if correct else "incorrect"
    print(f"Filtered {label} samples: {len(df)} -> {len(filtered_df)}")
    return filtered_df


def convert_relation(df: pd.DataFrame, old_relation: str, new_relation: str) -> pd.DataFrame:
    """
    Changes the relation name and updates the prompt text accordingly in a copy.

    Args:
        df: Input DataFrame
        old_relation: The original relation name to replace
        new_relation: The new relation name to set
    Returns:
        Updated DataFrame copy
    """
    df = df.copy()
    df["relation"] = new_relation
    df["prompt"] = df["prompt"].apply(lambda x: x.replace(f"Relation: {old_relation}", f"Relation: {new_relation}"))
    return df


def split_correct_samples(
    df: pd.DataFrame, 
    n_clean: int = 100, 
    n_corrupted: int = 25, 
    old_rel: str = "antonym", 
    new_rel: str = "none"
) -> Dict[str, pd.DataFrame]:
    """
    Splits the data into clean (converted) and corrupted sets using random sampling.

    Args:
        df: Input DataFrame of correct samples
        n_clean: Number of samples for the clean set
        n_corrupted: Number of samples for the corrupted set
        old_rel: Original relation name
        new_rel: New relation name for the clean set
    Returns:
        Dictionary containing 'clean' and 'corrupted' DataFrames
    """
    # Use pandas .sample() for safe and efficient sampling
    clean_samples = df.sample(n=min(n_clean, len(df)))
    
    remaining_df = df.drop(clean_samples.index)
    corrupted_samples = remaining_df.sample(n=min(n_corrupted, len(remaining_df)))
    
    # Convert relation for clean samples
    corrupted_samples = convert_relation(corrupted_samples, old_rel, new_rel)
    
    return clean_samples, corrupted_samples


def cache_activations(
    model: Any,
    df: pd.DataFrame,
) -> torch.Tensor:
    """
    Extracts attention output for the last token across all layers and heads, then averages them.
    The output tensor has shape (num_layers, num_heads, head_dim).

    Args:
        model: The language model (nnsight.LanguageModel)
        df: DataFrame containing prompts
    Returns:
        Averaged attention output tensor of shape (num_layers, num_heads, head_dim)
    """
    layer_stack = get_layer_stack(model)
    num_layers = len(layer_stack)
    num_heads = model.model.config.num_attention_heads
    hidden_size = model.model.config.hidden_size
    head_dim = hidden_size // num_heads

    all_attn_outs = []

    print(f"Extracting attention outputs for {len(df)} prompts...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Attention"):
        prompt = row["prompt"]
        
        # Dictionary to store saved nodes for each layer
        layer_saved_nodes = {}
        
        with model.trace(prompt, invoker_args={"truncation": False}):
            for layer_idx in range(num_layers):
                # Accessing attention output before the final linear projection (o_proj)
                # This typically contains head-separated information
                attn_out = layer_stack[layer_idx].self_attn.o_proj.input[:, -1, :].save()
                layer_saved_nodes[layer_idx] = attn_out

        # Extract and stack layers for this prompt: (num_layers, hidden_size)
        prompt_layers = []
        for layer_idx in range(num_layers):
            val = _resolve_saved_value(layer_saved_nodes[layer_idx]).detach().cpu()
            prompt_layers.append(val)
        
        # Shape: (num_layers, hidden_size)
        prompt_tensor = torch.stack(prompt_layers).squeeze(1)
        
        # Reshape to separate heads: (num_layers, num_heads, head_dim)
        prompt_tensor = prompt_tensor.view(num_layers, num_heads, head_dim)
        all_attn_outs.append(prompt_tensor)

    # Stack all prompts: (num_samples, num_layers, num_heads, head_dim)
    stacked_outs = torch.stack(all_attn_outs)
    
    # Average across samples (0th dimension)
    mean_attn_out = stacked_outs.mean(dim=0)
    
    return mean_attn_out


def trace_head_intervention_logit(
    model: Any,
    prompt: str,
    layer_idx: int,
    head_idx: int,
    head_dim: int,
    intervention_vector: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Runs inference with or without head-specific intervention on a single head.

    Args:
        model: The language model (nnsight.LanguageModel)
        prompt: Input prompt string
        layer_idx: Target layer index for intervention
        head_idx: Target head index for intervention
        head_dim: Dimension size of each attention head
        intervention_vector: Vector to inject into the head. If None, runs without intervention.
    Returns:
        Logits tensor for the last token position
    """
    layer_stack = get_layer_stack(model)
    out_proj = get_attn_out_proj_module(layer_stack[layer_idx])
    
    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim

    with model.trace(prompt, invoker_args={"truncation": False}):
        if intervention_vector is not None:
            # Inject the vector into the specific head's input at the last token
            vector = intervention_vector.to(device=model.device, dtype=model.dtype)
            out_proj.input[:, -1, start_idx:end_idx] = vector
        
        output_logits = model.output.logits[:, -1, :].save()

    return _resolve_saved_value(output_logits).detach().cpu().squeeze(0)


def compute_head_intervention_scores(
    model: Any,
    df: pd.DataFrame,
    mean_attn_outs: torch.Tensor,
) -> torch.Tensor:
    """
    Computes intervention scores for each layer and head by injecting mean vectors.

    Args:
        model: The language model (nnsight.LanguageModel)
        df: DataFrame containing prompts and labels
        mean_attn_outs: Mean attention output vectors of shape (num_layers, num_heads, head_dim)
    Returns:
        Intervention scores tensor of shape (num_layers, num_heads)
    """
    num_layers, num_heads, head_dim = mean_attn_outs.shape
    all_scores = []

    

    print(f"Computing intervention scores for {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Samples"):
        prompt = row["prompt"]
        output_id = row["output_id"]
        
        # Shape: (num_layers, num_heads)
        sample_scores = torch.zeros((num_layers, num_heads))
        
        # Get baseline logit (no intervention) once per sample
        baseline_logits = trace_head_intervention_logit(model, prompt, 0, 0, head_dim, None)
        baseline_val = baseline_logits[output_id].item()

        for l in tqdm(range(num_layers), desc="Layers", leave=False):
            for h in tqdm(range(num_heads), desc="Heads", leave=False):
                # Intervention with the corresponding mean vector
                intv_vector = mean_attn_outs[l, h]
                intv_logits = trace_head_intervention_logit(model, prompt, l, h, head_dim, intv_vector)
                intv_val = intv_logits[output_id].item()
                
                # Score: Intervention Logit - Baseline Logit
                sample_scores[l, h] = intv_val - baseline_val
        
        all_scores.append(sample_scores)

    # Stack all samples and compute mean across samples (0th dimension)
    final_scores = torch.stack(all_scores).mean(dim=0)
    
    return final_scores


def get_layer_stack(model):
    """
    Resolves the transformer layer stack from a model, supporting multiple architectures.

    Args:
        model: The language model
    Returns:
        The sequential container of transformer layers
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Could not find transformer layer stack on this model.")


def get_attn_out_proj_module(layer):
    """
    Resolves the attention output projection module from a transformer layer.

    Args:
        layer: A single transformer layer
    Returns:
        The output projection module (e.g. o_proj, out_proj, c_proj)
    """
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
        if hasattr(attn, "o_proj"):
            return attn.o_proj
        if hasattr(attn, "out_proj"):
            return attn.out_proj

    if hasattr(layer, "attn"):
        attn = layer.attn
        if hasattr(attn, "c_proj"):
            return attn.c_proj
        if hasattr(attn, "out_proj"):
            return attn.out_proj

    raise ValueError(f"Could not find attention output projection module for layer: {type(layer)}")


def _resolve_saved_value(saved_obj):
    """
    Unwraps an nnsight saved proxy object into its underlying tensor value.

    Args:
        saved_obj: Saved proxy object or raw tensor
    Returns:
        The unwrapped tensor value
    """
    return saved_obj.value if hasattr(saved_obj, "value") else saved_obj

def filter_top_heads(scores: torch.Tensor, top_k: int = 10) -> List[Tuple[int, int]]:
    """
    Selects top-k heads from a 2D scores tensor of shape (num_layers, num_heads).

    Args:
        scores: Intervention scores tensor of shape (num_layers, num_heads)
        top_k: Number of top heads to select
    Returns:
        List of (layer_idx, head_idx) tuples sorted by score descending
    """
    flat_indices = torch.topk(scores.flatten(), top_k).indices
    layers, heads = torch.unravel_index(flat_indices, scores.shape)
    return list(zip(layers.tolist(), heads.tolist()))


def build_intervention_vector(
    model: Any,
    top_heads: List[Tuple[int, int]],
    mean_attn_outs: torch.Tensor,
) -> torch.Tensor:
    """
    Builds a single intervention vector by projecting each top head's mean vector
    through its corresponding output projection and summing them.

    Args:
        model: The language model
        top_heads: List of (layer_idx, head_idx) tuples for top heads
        mean_attn_outs: Mean attention outputs of shape (num_layers, num_heads, head_dim)
    Returns:
        Intervention vector of shape (hidden_dim,)
    """
    layer_stack = get_layer_stack(model)
    num_heads = model.model.config.num_attention_heads
    hidden_size = model.model.config.hidden_size
    head_dim = hidden_size // num_heads

    intervention_vector = torch.zeros(hidden_size)

    for layer_idx, head_idx in top_heads:
        head_vector = mean_attn_outs[layer_idx, head_idx]

        o_weight = get_attn_out_proj_module(layer_stack[layer_idx]).weight.data
        head_o_weight = o_weight[:, head_idx * head_dim : (head_idx + 1) * head_dim]

        projected = head_vector.to(device=head_o_weight.device, dtype=head_o_weight.dtype) @ head_o_weight.t()
        intervention_vector += projected.detach().cpu()

    return intervention_vector

