# TTS Training Pipeline

The training pipeline is run with the `training-pipeline` console command created by
`uv`.

## Configuration

Pass a YAML file containing the dataset, model, and training settings:

```yaml
training:
  batch_size: 32
  eval_batch_size: 4
  epochs: 10
  learning_rate: 0.001
  output_path: outputs/demo
  eval_split_size: 0.1
  eval_split_max_size: 100
  warm_start:
    enabled: false
    checkpoint_path: ""

dataset:
  path: data/train.csv
  formatter: default
  meta_file_train: data/metadata.csv

model:
  type: tacotron2
  sample_rate: 22050
```

Set `warm_start.enabled` to `true` and provide an existing model checkpoint in
`warm_start.checkpoint_path` to fine-tune that checkpoint. Fresh runs leave
`restore_path` empty and create a new training run.

## Run

```bash
uv run training-pipeline config.yaml
```

The command is also available after installing the package with `uv sync`:

```bash
uv run python main.py config.yaml
```