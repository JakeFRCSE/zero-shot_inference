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


def convert_relation(
    df: pd.DataFrame,
    new_relation: str,
    prompt_template: str = "Relation: {relation}\nInput: {input}\nOutput:",
) -> pd.DataFrame:
    """
    Changes the relation name and regenerates prompts from the template.

    Args:
        df: Input DataFrame with 'input' and 'relation' columns
        new_relation: The new relation name to set
        prompt_template: Format string with {relation}, {input}, {output} placeholders
    Returns:
        Updated DataFrame copy
    """
    df = df.copy()
    df["relation"] = new_relation
    df["prompt"] = df.apply(
        lambda row: prompt_template.format(
            relation=new_relation, input=row["input"], output="",
        ).strip(),
        axis=1,
    )
    return df


def split_correct_samples(
    df: pd.DataFrame, 
    n_clean: int = 100, 
    n_corrupted: int = 25, 
    new_rel: str = "none",
    prompt_template: str = "Relation: {relation}\nInput: {input}\nOutput:",
) -> Dict[str, pd.DataFrame]:
    """
    Splits the data into clean (converted) and corrupted sets using random sampling.

    Args:
        df: Input DataFrame of correct samples
        n_clean: Number of samples for the clean set
        n_corrupted: Number of samples for the corrupted set
        new_rel: New relation name for the corrupted set
        prompt_template: Format string with {relation}, {input}, {output} placeholders
    Returns:
        Dictionary containing 'clean' and 'corrupted' DataFrames
    """
    clean_samples = df.sample(n=min(n_clean, len(df)))
    
    remaining_df = df.drop(clean_samples.index)
    if len(remaining_df) == 0:
        raise ValueError(
            f"No samples left for corrupted set (total={len(df)}, n_clean={n_clean}). "
            f"Reduce n_clean or use more data."
        )
    corrupted_samples = remaining_df.sample(n=min(n_corrupted, len(remaining_df)))
    
    corrupted_samples = convert_relation(corrupted_samples, new_rel, prompt_template=prompt_template)
    
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
    cfg = get_model_config(model)
    num_heads = cfg.num_attention_heads
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // num_heads

    out_projs = [get_attn_out_proj_module(layer_stack[i]) for i in range(num_layers)]
    all_attn_outs = []

    print(f"Extracting attention outputs for {len(df)} prompts...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Attention"):
        prompt = row["prompt"]
        
        with model.trace(prompt, invoker_args={"truncation": False}):
            layer_saved_nodes = list().save()
            for layer_idx in range(num_layers):
                layer_saved_nodes.append(out_projs[layer_idx].input[:, -1, :])

        prompt_layers = [
            _resolve_saved_value(layer_saved_nodes[i]).detach().cpu()
            for i in range(num_layers)
        ]
        
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


_BACKBONE_ATTRS = ("model", "gpt_neox", "transformer")


def _get_backbone(model):
    """
    Resolves the inner backbone module from a causal LM wrapper.

    Args:
        model: The language model
    Returns:
        The backbone module (e.g. LlamaModel, GPTNeoXModel, GPT2Model)
    """
    for attr in _BACKBONE_ATTRS:
        backbone = getattr(model, attr, None)
        if backbone is not None:
            return backbone
    raise ValueError("Could not resolve backbone module on this model.")


def get_model_config(model):
    """
    Retrieves the model config, supporting multiple architectures
    including nnsight LanguageModel wrappers.

    Args:
        model: The language model (HuggingFace or nnsight wrapper)
    Returns:
        The model's configuration object
    """
    config = getattr(model, "config", None)
    if config is not None:
        return config

    inner = getattr(model, "_model", None) or getattr(model, "model", None)
    if inner is not None and inner is not model:
        config = getattr(inner, "config", None)
        if config is not None:
            return config

    return _get_backbone(model).config


def get_layer_stack(model):
    """
    Resolves the transformer layer stack from a model, supporting multiple architectures.

    Args:
        model: The language model
    Returns:
        The sequential container of transformer layers
    """
    _LAYER_CHILD = {"model": "layers", "gpt_neox": "layers", "transformer": "h"}
    for attr in _BACKBONE_ATTRS:
        backbone = getattr(model, attr, None)
        if backbone is not None:
            child = _LAYER_CHILD.get(attr)
            layers = getattr(backbone, child, None) if child else None
            if layers is not None:
                return layers
    raise ValueError("Could not find transformer layer stack on this model.")


_ATTN_OUT_PROJ_PATHS = [
    ("self_attn", "o_proj"),    # LLaMA, Qwen
    ("self_attn", "out_proj"),  # some HF variants
    ("attention", "dense"),     # Pythia / GPT-NeoX
    ("attn", "c_proj"),         # GPT-2
    ("attn", "out_proj"),       # some HF variants
]


def get_attn_out_proj_module(layer):
    """
    Resolves the attention output projection module from a transformer layer.

    Args:
        layer: A single transformer layer
    Returns:
        The output projection module (e.g. o_proj, dense, c_proj)
    """
    for attn_attr, proj_attr in _ATTN_OUT_PROJ_PATHS:
        attn = getattr(layer, attn_attr, None)
        if attn is not None:
            proj = getattr(attn, proj_attr, None)
            if proj is not None:
                return proj
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
    cfg = get_model_config(model)
    num_heads = cfg.num_attention_heads
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // num_heads

    intervention_vector = torch.zeros(hidden_size)

    for layer_idx, head_idx in top_heads:
        head_vector = mean_attn_outs[layer_idx, head_idx]

        o_weight = get_attn_out_proj_module(layer_stack[layer_idx]).weight.data
        head_o_weight = o_weight[:, head_idx * head_dim : (head_idx + 1) * head_dim]

        projected = head_vector.to(device=head_o_weight.device, dtype=head_o_weight.dtype) @ head_o_weight.t()
        intervention_vector += projected.detach().cpu()

    return intervention_vector

