import os
import csv

from training_pipeline.config_manager import DatasetConfig


def load_dataset(config: DatasetConfig):
    metadata_path = os.path.join(config.path, config.meta_file_train)
    audio_dir = os.path.join(config.path, "audio")

    samples = []

    if not os.path.exists(metadata_path):
        raise ValueError(f"{config.meta_file_train} not found")

    with open(metadata_path, "r") as f:
        reader = csv.DictReader(f, delimiter="|")
        for i, row in enumerate(reader):
            if i >= config.max_samples:
                break

            audio_path = os.path.join(audio_dir, row["audio_file"])

            if not os.path.exists(audio_path):
                raise ValueError(f"Missing audio file: {row['audio_file']}")

            samples.append({
                "audio": audio_path,
                "text": row["text"]
            })

    return samples

    
def preprocess_text(samples):
    for sample in samples:
        sample["text"] = sample["text"].strip().lower()
    return samples

    
# This should run BEFORE preprocessing, so that we don't get errors from 
# the preprocessing step. It should also run before loading the dataset    
def validate_dataset(samples):
    errors = []
    for sample in samples:
        audio = sample.get("audio")

        if isinstance(audio, str):
            if not os.path.exists(audio):
                errors.append(f"Missing audio file: {audio}")
        elif audio is None:
            errors.append("Missing audio file")

        text = sample.get("text")
        if not text or not str(text).strip():
            errors.append("Text is empty")
    return errors