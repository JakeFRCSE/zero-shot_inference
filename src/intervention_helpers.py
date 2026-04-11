from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

from patching_helpers import get_layer_stack, _resolve_saved_value


def _trace_layer_intervention_logit(
    model: Any,
    prompt: str,
    layer_idx: Optional[int],
    intervention_vector: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Runs inference on a single prompt, optionally adding the intervention vector
    to the hidden state of the last token at the specified layer.

    Args:
        model: The language model (nnsight.LanguageModel)
        prompt: Input prompt string
        layer_idx: Target layer index for intervention (None for baseline)
        intervention_vector: Vector to add at the target layer (None for baseline)
    Returns:
        Logits tensor for the last token (torch.Tensor)
    """
    layer_stack = get_layer_stack(model)

    with model.trace(prompt, invoker_args={"truncation": False}):
        if layer_idx is not None and intervention_vector is not None:
            vector = intervention_vector.to(device=model.device, dtype=model.dtype)
            hidden = layer_stack[layer_idx].output[0]
            if hidden.ndim == 3:
                hidden[:, -1, :] += vector
            else:
                hidden += vector

        output_logits = model.output.logits[:, -1, :].save()

    return _resolve_saved_value(output_logits).detach().cpu().squeeze(0)


def evaluate_layer_intervention(
    model: Any,
    df: pd.DataFrame,
    intervention_vector: torch.Tensor,
) -> pd.DataFrame:
    """
    Evaluates the effect of adding an intervention vector at each layer.
    For every prompt, runs a baseline (no intervention) and one intervention per layer.

    Args:
        model: The language model (nnsight.LanguageModel)
        df: DataFrame with columns [input, output, relation, prompt, input_id, output_id, relation_id]
        intervention_vector: Vector of shape (hidden_dim,) to add at each layer
    Returns:
        DataFrame with intervention results for all prompts and layers
    """
    num_layers = len(get_layer_stack(model))
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Prompts"):
        prompt = row["prompt"]

        for layer_idx in [None] + list(range(num_layers)):
            logits = _trace_layer_intervention_logit(model, prompt, layer_idx, intervention_vector)
            prediction_id = int(logits.argmax().item())

            results.append({
                "input": row["input"],
                "output": row["output"],
                "relation": row["relation"],
                "prompt": prompt,
                "intervention_layer": layer_idx,
                "input_id": row["input_id"],
                "output_id": row["output_id"],
                "relation_id": row["relation_id"],
                "prediction_id": prediction_id,
                "input_logit": logits[row["input_id"]].item(),
                "output_logit": logits[row["output_id"]].item(),
                "relation_logit": logits[row["relation_id"]].item(),
                "prediction_logit": logits[prediction_id].item(),
                "input_prediction": prediction_id == row["input_id"],
                "output_prediction": prediction_id == row["output_id"],
                "relation_prediction": prediction_id == row["relation_id"],
            })

    return pd.DataFrame(results)
