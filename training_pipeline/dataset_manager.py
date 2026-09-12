import os
import csv
import soundfile as sf
import pyloudnorm as pyln

class DatasetManager:
    @staticmethod
    def load_dataset(path: str, max_samples: int):
        metadata_path = os.path.join(path, "metadata.csv")
        audio_dir = os.path.join(path, "audio")

        samples = []

        if not os.path.exists(metadata_path):
            raise ValueError("metadata.csv not found")

        with open(metadata_path, "r") as f:
            reader = csv.DictReader(f, delimiter="|")
            for i, row in enumerate(reader):
                if i >= max_samples:
                    break

                audio_path = os.path.join(audio_dir, row["audio_file"])

                if not os.path.exists(audio_path):
                    raise ValueError(f"Missing audio file: {row['audio_file']}")

                samples.append({
                    "audio": audio_path,
                    "text": row["text"]
                })

        return samples

    @staticmethod
    def preprocess_dataset(samples):
        processed_samples = []
        for sample in samples:
            text = sample["text"].strip().lower()

            audio, rate = sf.read(sample["audio"])

            meter = pyln.Meter(rate)
            loudness = meter.integrated_loudness(audio)
            #loudness normalized data 
            sample["audio"] = pyln.normalize.loudness(audio, loudness, -23.0)

            processed_samples.append({
                #should we have a field for the original audio path? and keep the normalized audio its own field?
                "audio": sample["audio"],
                "text": text
                #should we have a field for loudness?
                #"loudness": loudness
                #should we have a field for sample rate?
                #"sample_rate": rate
            })
        return processed_samples
    
    # This should run BEFORE preprocessing, so that we don't get errors from 
    # the preprocessing step. It should also run before loading the dataset
    @staticmethod
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