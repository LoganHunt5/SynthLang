import pytest

from training_pipeline import dataset_manager
from training_pipeline.config_manager import DatasetConfig


def create_dataset(tmp_path, metadata, audio_files=()):
    dataset_dir = tmp_path / "dataset"
    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir(parents=True)

    (dataset_dir / "metadata.csv").write_text(metadata)
    for audio_file in audio_files:
        (audio_dir / audio_file).touch()

    return dataset_dir


def make_config(path, max_samples=10, meta_file_train="metadata.csv"):
    return DatasetConfig(
        path=str(path),
        max_samples=max_samples,
        formatter="default",
        meta_file_train=meta_file_train,
    )


def test_load_dataset_reads_pipe_delimited_metadata_and_audio_paths(tmp_path):
    dataset_dir = create_dataset(
        tmp_path,
        "audio_file|text\nhello.wav| Hello there! \n",
        audio_files=("hello.wav",),
    )

    result = dataset_manager.load_dataset(make_config(dataset_dir, max_samples=10))

    assert result == [
        {"audio": str(dataset_dir / "audio" / "hello.wav"), "text": " Hello there! "}
    ]


def test_load_dataset_limits_number_of_samples(tmp_path):
    dataset_dir = create_dataset(
        tmp_path,
        "audio_file|text\nfirst.wav|First\nsecond.wav|Second\n",
        audio_files=("first.wav", "second.wav"),
    )

    result = dataset_manager.load_dataset(make_config(dataset_dir, max_samples=1))

    assert result == [
        {"audio": str(dataset_dir / "audio" / "first.wav"), "text": "First"}
    ]


def test_load_dataset_raises_when_metadata_is_missing(tmp_path):
    with pytest.raises(ValueError, match="metadata.csv not found"):
        dataset_manager.load_dataset(make_config(tmp_path, max_samples=1))


def test_load_dataset_raises_when_audio_is_missing(tmp_path):
    dataset_dir = create_dataset(
        tmp_path,
        "audio_file|text\nmissing.wav|Missing audio\n",
    )

    with pytest.raises(ValueError, match="Missing audio file: missing.wav"):
        dataset_manager.load_dataset(make_config(dataset_dir, max_samples=1))


def test_preprocess_text_strips_and_lowercases_text_in_place():
    samples = [{"audio": "sample.wav", "text": "  Hello WORLD  "}]

    result = dataset_manager.preprocess_text(samples)

    assert result is samples
    assert samples == [{"audio": "sample.wav", "text": "hello world"}]


def test_validate_dataset_returns_no_errors_for_valid_samples(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()
    samples = [{"audio": str(audio_path), "text": "Transcript"}]

    assert dataset_manager.validate_dataset(samples) == []


def test_validate_dataset_reports_missing_audio_and_empty_text(tmp_path):
    existing_audio = tmp_path / "existing.wav"
    existing_audio.touch()
    samples = [
        {"audio": str(tmp_path / "missing.wav"), "text": "Transcript"},
        {"audio": None, "text": " "},
        {"audio": str(existing_audio), "text": ""},
    ]

    assert dataset_manager.validate_dataset(samples) == [
        f"Missing audio file: {tmp_path / 'missing.wav'}",
        "Missing audio file",
        "Text is empty",
        "Text is empty",
    ]


def test_validate_dataset_ignores_non_string_audio_values():
    samples = [{"audio": b"audio data", "text": "Transcript"}]

    assert dataset_manager.validate_dataset(samples) == []