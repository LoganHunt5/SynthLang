# TTS Training Pipeline

The training pipeline is run with the `training-pipeline` console command created by
`uv`.

## Configuration

Pass a YAML file containing the dataset, model, and training settings:

```yaml
backend: coqui            # optional, defaults to coqui

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
  type: tacotron2         # optional, defaults to tacotron2
  sample_rate: 22050
  params:                 # optional backend/architecture specific settings
    r: 2
    attention_type: dynamic_convolution
```

`backend` selects the framework that trains the model and `model.type` selects
the architecture within that backend. Supported combinations:

| backend | model.type |
|---------|------------|
| coqui   | tacotron2  |

The `training`, `dataset`, and `model.sample_rate` settings are generic: every
backend honors them. `model.params` is passed to the selected backend as-is, so
it holds whatever settings that architecture understands (for Coqui, any field
of the model's config class, such as `Tacotron2Config`). Values in `params`
override both the architecture defaults and the generic settings, and an
unrecognized key is rejected with an error. For Coqui, `training.learning_rate`
is passed through as the model config's `lr`.

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