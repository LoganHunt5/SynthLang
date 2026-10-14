import os
import csv

from training_pipeline.pipeline.config_manager import DatasetConfig


def load_dataset(config: DatasetConfig):
    audio_dir = os.path.join(config.path, "audio")

    errors, rows = _validate_dataset(config)
    if errors:
        raise ValueError("; ".join(errors))

    samples = []
    for row in rows:
        audio_path = os.path.join(audio_dir, row["audio_file"])

        samples.append({
            "audio": audio_path,
            "text": row["text"]
        })

    return samples

    
def preprocess_text(samples):
    for sample in samples:
        sample["text"] = sample["text"].strip().lower()
    return samples

    
# This is to be called by the load function
def _validate_dataset(config: DatasetConfig):
    errors = []
    rows = []
    metadata_path = os.path.join(config.path, config.meta_file_train)
    audio_dir = os.path.join(config.path, "audio")

    if not os.path.exists(metadata_path):
        return [f"{config.meta_file_train} not found"], rows

    with open(metadata_path, "r") as f:
        reader = csv.DictReader(f, delimiter="|")

        if not {"audio_file", "text"}.issubset(reader.fieldnames or []):
            return ["Metadata must contain audio_file and text columns"], rows

        for row in reader:
            rows.append(row)
            audio = row.get("audio_file")
            audio_path = os.path.join(audio_dir, audio) if audio else None

            if audio_path is None:
                errors.append("Missing audio file")
            elif not os.path.exists(audio_path):
                errors.append(f"Missing audio file: {audio}")

            text = row.get("text")
            if not text or not str(text).strip():
                errors.append("Text is empty")
    return errors, rows