from pathlib import Path
import pandas as pd

RESULTS_DIR = Path('results')

METRICS_FILES = {
    'normal': 'layer_metrics.csv',
    'restore': 'layer_restore_metrics.csv',
    'enforce': 'layer_enforce_metrics.csv',
}
METRIC_COLUMNS = ['input_prediction', 'output_prediction', 'relation_prediction']

tables = {kind: [] for kind in METRICS_FILES}

for prompt_dir in sorted(RESULTS_DIR.glob('*/*/prompt_*')):
    model = prompt_dir.parent.parent.name
    dataset = prompt_dir.parent.name
    prompt = prompt_dir.name

    for kind, filename in METRICS_FILES.items():
        path = prompt_dir / filename
        row = {'model': model, 'dataset': dataset, 'prompt': prompt}

        baseline_cols = [f'baseline_{col}' for col in METRIC_COLUMNS]
        if not path.exists():
            row.update({col: None for col in ['num_layers', 'target_layer'] + baseline_cols + METRIC_COLUMNS})
            tables[kind].append(row)
            continue

        df = pd.read_csv(path)
        num_layers = int(df['intervention_layer'].dropna().max()) + 1
        target_layer = num_layers // 3

        baseline_row = df[df['intervention_layer'].isna()]
        layer_row = df[df['intervention_layer'] == target_layer]
        row['num_layers'] = num_layers
        row['target_layer'] = target_layer
        for col in METRIC_COLUMNS:
            row[f'baseline_{col}'] = baseline_row[col].iloc[0] if not baseline_row.empty else None
            row[col] = layer_row[col].iloc[0] if not layer_row.empty else None

        tables[kind].append(row)

for kind, rows in tables.items():
    out = pd.DataFrame(rows)
    save_path = RESULTS_DIR / f'analysis_{kind}.csv'
    out.to_csv(save_path, index=False)
