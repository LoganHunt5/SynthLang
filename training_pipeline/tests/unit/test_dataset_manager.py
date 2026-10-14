import tempfile
import unittest
from pathlib import Path

from training_pipeline.pipeline import dataset_manager
from training_pipeline.pipeline.config_manager import DatasetConfig


def create_dataset(tmp_path, metadata, audio_files=()):
    dataset_dir = tmp_path / "dataset"
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir(parents=True)

    (dataset_dir / "metadata.csv").write_text(metadata)
    for audio_file in audio_files:
        (audio_dir / audio_file).touch()

    return dataset_dir


def make_config(path, meta_file_train="metadata.csv"):
    return DatasetConfig(
        path=str(path),
        formatter="default",
        meta_file_train=meta_file_train,
    )


class TestDatasetManager(unittest.TestCase):
    def test_load_dataset_reads_pipe_delimited_metadata_and_audio_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = create_dataset(
                Path(tmp_dir),
                "audio_file|text\nhello.wav| Hello there! \n",
                audio_files=("hello.wav",),
            )

            result = dataset_manager.load_dataset(make_config(dataset_dir))

            self.assertEqual(
                result,
                [{"audio": str(dataset_dir / "audio" / "hello.wav"), "text": " Hello there! "}],
            )

    # tests multiple metadata rows and ordering
    def test_load_dataset_preserves_multiple_rows_in_metadata_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = create_dataset(
                Path(tmp_dir),
                "audio_file|text\nfirst.wav|First\nsecond.wav|Second\n",
                audio_files=("first.wav", "second.wav"),
            )

            result = dataset_manager.load_dataset(make_config(dataset_dir))

            self.assertEqual(
                result,
                [
                    {"audio": str(dataset_dir / "audio" / "first.wav"), "text": "First"},
                    {"audio": str(dataset_dir / "audio" / "second.wav"), "text": "Second"},
                ],
            )
    # tests custom metadata filenames
    def test_load_dataset_accepts_custom_metadata_filename(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = create_dataset(
                Path(tmp_dir),
                "audio_file|text\nsample.wav|Transcript\n",
                audio_files=("sample.wav",),
            )
            custom_metadata = dataset_dir / "metadata.csv"
            custom_metadata.rename(dataset_dir / "train_metadata.csv")

            result = dataset_manager.load_dataset(
                make_config(dataset_dir, meta_file_train="train_metadata.csv")
            )

            self.assertEqual(result[0]["text"], "Transcript")
    
    # tests header-only metadata
    def test_load_dataset_accepts_header_only_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = create_dataset(Path(tmp_dir), "audio_file|text\n")

            self.assertEqual(dataset_manager.load_dataset(make_config(dataset_dir)), [])

    # tests missing meta data
    def test_load_dataset_raises_when_metadata_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "metadata.csv not found"):
                dataset_manager.load_dataset(make_config(Path(tmp_dir)))

    # tests missing audio files
    def test_load_dataset_raises_when_audio_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = create_dataset(
                Path(tmp_dir),
                "audio_file|text\nmissing.wav|Missing audio\n",
            )

            with self.assertRaisesRegex(ValueError, "Missing audio file: missing.wav"):
                dataset_manager.load_dataset(make_config(dataset_dir))

    # tests if the entire audio directory is missing
    def test_load_dataset_rejects_audio_file_when_audio_directory_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "metadata.csv").write_text(
                "audio_file|text\nsample.wav|Transcript\n"
            )

            with self.assertRaisesRegex(ValueError, "Missing audio file: sample.wav"):
                dataset_manager.load_dataset(make_config(dataset_dir))

    def test_preprocess_text_strips_and_lowercases_text_in_place(self):
        samples = [{"audio": "sample.wav", "text": "  Hello WORLD  "}]

        result = dataset_manager.preprocess_text(samples)

        self.assertIs(result, samples)
        self.assertEqual(samples, [{"audio": "sample.wav", "text": "hello world"}])

    # tests missing audio and empty text
    def test_load_dataset_rejects_missing_audio_and_empty_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = create_dataset(
                Path(tmp_dir),
                "audio_file|text\nmissing.wav|Transcript\n| \nexisting.wav|\n",
                audio_files=("existing.wav",),
            )

            with self.assertRaisesRegex(
                ValueError,
                "Missing audio file: missing.wav; Missing audio file; Text is empty",
            ):
                dataset_manager.load_dataset(make_config(dataset_dir))

    # tests missing metadata columns
    def test_load_dataset_rejects_missing_metadata_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = create_dataset(Path(tmp_dir), "audio_file|label\na.wav|x\n")

            with self.assertRaisesRegex(
                ValueError,
                "Metadata must contain audio_file and text columns",
            ):
                dataset_manager.load_dataset(make_config(dataset_dir))


if __name__ == "__main__":
    unittest.main()