from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from io_helpers import load_file, save_file

DATASET_DIR: Optional[Path] = None

DEFAULT_PROMPT_TEMPLATE = "Relation: {relation}\nInput: {input}\nOutput: {output}"
ALTERNATIVE_PROMPT_TEMPLATE = "Q: What is the {relation} of {input}?\nA: {output}"


def resolve_device(device: Optional[str] = None) -> str:
    """
    Returns the optimal computing device (CUDA or CPU) for inference.
    Prioritizes the user-specified device if provided.

    Args:
        device: Explicitly specified device name (Optional)
    Returns:
        The device name to be used (str)
    """
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_prompt(
    relation: str,
    item: Dict,
    include_output: bool = False,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> str:
    """
    Generates a model input prompt based on the given relation and item data.
    The inclusion of the target output can be toggled for different use cases.

    Args:
        relation: Description of the relationship between data points
        item: Dictionary containing 'input' and 'output' values
        include_output: Whether to include the target output in the prompt
        prompt_template: Format string with {relation}, {input}, {output} placeholders
    Returns:
        The generated prompt string (str)
    """
    input_val = item.get("input", "")
    target_output = item.get("output", "")
    display_output = target_output if include_output else ""
    prompt = prompt_template.format(
        relation=relation, input=input_val, output=display_output,
    )
    return prompt.strip()


def load_hf_model_and_tokenizer(model_id: str = "gpt2"):
    """
    Loads a pre-trained model and tokenizer from Hugging Face for a given model ID.
    Configures appropriate data types and device mapping based on GPU availability.

    Args:
        model_id: Hugging Face model identifier to load
    Returns:
        A tuple of the loaded model and tokenizer (model, tokenizer)
    """
    print(f"Loading model and tokenizer for: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, tokenizer


def get_first_token_id(tokenizer, text: str, add_leading_space: bool = True) -> int:
    """
    Tokenizes the given text and returns the ID of the first token.
    Allows adjusting tokenization results by optionally adding a leading space.

    Args:
        tokenizer: Tokenizer to use for processing text
        text: Target string to tokenize
        add_leading_space: Whether to prepend a space to the text
    Returns:
        The ID of the first token (int)
    """
    text = text.strip()
    
    prefix = " " if add_leading_space else ""
    formatted_text = prefix + text
    
    input_ids = tokenizer.encode(formatted_text, add_special_tokens=False)
    
    if not input_ids:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
    
    if input_ids:
        return input_ids[0]
        
    raise ValueError(f"Could not tokenize text: {text!r}")


def token_ids_match(tokenizer, id_a: int, id_b: int) -> bool:
    """
    Compares two token IDs by decoding and normalizing their text
    (strip whitespace + lowercase).

    Args:
        tokenizer: Tokenizer for decoding token IDs
        id_a: First token ID
        id_b: Second token ID
    Returns:
        True if the normalized decoded texts are equal
    """
    if id_a == id_b:
        return True
    text_a = tokenizer.decode([id_a], skip_special_tokens=True).strip().lower()
    text_b = tokenizer.decode([id_b], skip_special_tokens=True).strip().lower()
    return text_a == text_b


def get_last_logits(model, tokenizer, prompt: str, device: Optional[str] = None) -> torch.Tensor:
    """
    Runs the model on the input prompt and extracts the logits for the last token.
    The extracted logits are moved to CPU memory before being returned.

    Args:
        model: Language model to use for inference
        tokenizer: Tokenizer for processing input text
        prompt: Input prompt string for the model
        device: Device to perform inference on (Optional)
    Returns:
        Logits tensor for the last token (torch.Tensor)
    """
    resolved_device = resolve_device(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(resolved_device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits[:, -1, :].detach().cpu().squeeze(0)


def get_predicted_id(logits: torch.Tensor) -> int:
    """
    Returns the ID of the token with the highest value in the given logits tensor.

    Args:
        logits: Model output logits tensor
    Returns:
        The ID of the token with the highest probability (int)
    """
    logits_1d = logits.detach().cpu().reshape(-1)
    return int(logits_1d.argmax().item())


def get_token_logit(logits: torch.Tensor, token_id: int) -> float:
    """
    Extracts the logit value for a specific token ID from the logits tensor.

    Args:
        logits: Model output logits tensor
        token_id: ID of the token to extract the logit for
    Returns:
        The logit value of the specified token (float)
    """
    return logits[token_id].item()


def evaluate_sample(
    model, tokenizer, relation: str, item: Dict, device: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Dict[str, object]:
    """
    Performs inference on a single data item and constructs a result dictionary.
    Includes predictions, ground truth comparisons, and logits for key tokens.

    Args:
        model: Language model for inference
        tokenizer: Tokenizer for text processing
        relation: Description of the relationship
        item: Data dictionary containing input and output information
        device: Device to perform inference on
        prompt_template: Format string with {relation}, {input}, {output} placeholders
    Returns:
        Dictionary containing inference results and metadata (Dict)
    """
    prompt = generate_prompt(relation, item, prompt_template=prompt_template)

    logits = get_last_logits(model, tokenizer, prompt, device=device)
    prediction = tokenizer.decode(get_predicted_id(logits), skip_special_tokens=True).strip()

    prediction_id = get_predicted_id(logits)
    input_id = get_first_token_id(tokenizer, item["input"])
    output_id = get_first_token_id(tokenizer, item["output"])
    relation_id = get_first_token_id(tokenizer, relation)

    input_logit = get_token_logit(logits, input_id)
    output_logit = get_token_logit(logits, output_id)
    relation_logit = get_token_logit(logits, relation_id)

    output_prediction = token_ids_match(tokenizer, prediction_id, output_id)
    input_prediction = token_ids_match(tokenizer, prediction_id, input_id)
    relation_prediction = token_ids_match(tokenizer, prediction_id, relation_id)
    
    return {
        "input": item["input"],
        "output": item["output"],
        "relation": relation,
        "prompt": prompt,
        "input_id": input_id,
        "output_id": output_id,
        "relation_id": relation_id,
        "prediction": prediction,
        "logits": logits,
        "prediction_id": prediction_id,
        "input_logit": input_logit,
        "output_logit": output_logit,
        "relation_logit": relation_logit,
        "input_prediction": input_prediction,
        "output_prediction": output_prediction,
        "relation_prediction": relation_prediction,
    }


def evaluate_relation(
    model, tokenizer, data: List[Dict], relation: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> pd.DataFrame:
    """
    Evaluates the model on a list of data items belonging to a single relation.
    Collects inference results for each sample into a Pandas DataFrame.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for text processing
        data: List of evaluation data items
        relation: Name of the relation being evaluated
        prompt_template: Format string with {relation}, {input}, {output} placeholders
    Returns:
        DataFrame containing all evaluation results (pd.DataFrame)
    """
    device = resolve_device()
    results = []

    print(f"Evaluating {len(data)} samples...")

    for item in tqdm(data, total=len(data), desc="Inference"):
        result_row = evaluate_sample(model, tokenizer, relation, item, device, prompt_template=prompt_template)
        results.append(result_row)

    results_df = pd.DataFrame(results)
    print(f"Evaluation complete.")
    return results_df


def evaluate_relations(
    model, tokenizer, data: List[Dict], relations: List[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluates the model across multiple relations sequentially.
    Stores and returns the evaluation results for each relation in a dictionary.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for text processing
        data: List of evaluation data items
        relations: List of relation names to evaluate
        prompt_template: Format string with {relation}, {input}, {output} placeholders
    Returns:
        Dictionary mapping relation names to their result DataFrames (Dict)
    """
    results = {}
    for relation in relations:
        results[relation] = evaluate_relation(model, tokenizer, data, relation, prompt_template=prompt_template)
    return results


def summarize_layer_metrics(
    intervention_results_df: pd.DataFrame,
    metrics: List[str] = ("input_prediction", "output_prediction", "relation_prediction"),
) -> pd.DataFrame:
    """
    Computes mean accuracy (%) per intervention layer, preserving the
    baseline (None / NaN) layer at the top of the resulting DataFrame.

    Args:
        intervention_results_df: DataFrame with an 'intervention_layer' column
            and boolean prediction columns.
        metrics: Columns to aggregate.
    Returns:
        DataFrame indexed by intervention_layer with mean accuracy in percent.
    """
    layer_metrics = (
        intervention_results_df
        .groupby("intervention_layer", dropna=False)[list(metrics)]
        .mean() * 100
    )

    numeric_layers = sorted(l for l in layer_metrics.index if l is not None and not pd.isna(l))
    ordered_index = [None] + numeric_layers
    return layer_metrics.reindex(ordered_index)


def run_evaluation(
    model,
    tokenizer,
    relation: str | List[str],
    load_path: Path,
    save_dir: Path,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> pd.DataFrame:
    """
    Executes the full inference pipeline: data loading, evaluation, and saving results.
    Supports evaluation for either a single relation or multiple relations.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for text processing
        relation: Single relation name or a list of relation names
        load_path: Path to load the input data from
        save_dir: Path to save the evaluation results to
        prompt_template: Format string with {relation}, {input}, {output} placeholders
    Returns:
        The final evaluation results (Dict[str, pd.DataFrame] or pd.DataFrame)
    """
    data = load_file(load_path)

    if isinstance(relation, str):
        results_df = evaluate_relation(model, tokenizer, data, relation, prompt_template=prompt_template)
    else:
        results_df = evaluate_relations(model, tokenizer, data, relation, prompt_template=prompt_template)

    if isinstance(results_df, pd.DataFrame):
        save_file(results_df, save_dir / f"_{relation}.csv")
    else:
        for relation, df in results_df.items():
            save_file(df, save_dir / f"_{relation}.csv")
    return results_df
