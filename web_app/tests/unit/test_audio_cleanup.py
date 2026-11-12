import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import main


class TestAudioCleanup(unittest.TestCase):
    def test_delete_old_audio_removes_stale_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            stale = audio_dir / "stale.wav"
            fresh = audio_dir / "fresh.mp3"
            stale.write_bytes(b"old")
            fresh.write_bytes(b"new")

            stale_ts = time.time() - main.AUDIO_TTL_SECONDS - 10
            os.utime(stale, (stale_ts, stale_ts))

            with patch.object(main, "AUDIO_DIR", audio_dir):
                main.delete_old_audio()

            self.assertFalse(stale.exists())
            self.assertTrue(fresh.exists())

    def test_delete_old_audio_keeps_recent_audio_and_ignores_non_audio_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            recent = audio_dir / "recent.wav"
            notes = audio_dir / "notes.txt"
            recent.write_bytes(b"new")
            notes.write_bytes(b"not audio")

            with patch.object(main, "AUDIO_DIR", audio_dir):
                main.delete_old_audio()

            self.assertTrue(recent.exists())
            self.assertTrue(notes.exists())
