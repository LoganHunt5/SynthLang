import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training_pipeline.pipeline.logger import configure_logger


class TestConfigureLogger(unittest.TestCase):
	def setUp(self):
		self.logger = logging.getLogger("training_pipeline")
		self.original_handlers = self.logger.handlers[:]
		self.original_level = self.logger.level
		self.original_propagate = self.logger.propagate

		for handler in self.original_handlers:
			self.logger.removeHandler(handler)

	def tearDown(self):
		for handler in self.logger.handlers[:]:
			self.logger.removeHandler(handler)
			handler.close()

		for handler in self.original_handlers:
			self.logger.addHandler(handler)
		self.logger.setLevel(self.original_level)
		self.logger.propagate = self.original_propagate

	def test_configures_console_handler_at_info_level(self):
		console_output = io.StringIO()
		with patch("sys.stderr", console_output):
			logger = configure_logger()
			logger.debug("debug message")
			logger.info("info message")

		self.assertIs(logger, self.logger)
		self.assertEqual(logger.level, logging.DEBUG)
		self.assertFalse(logger.propagate)
		self.assertEqual(len(logger.handlers), 1)
		self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
		self.assertEqual(logger.handlers[0].level, logging.INFO)
		self.assertNotIn("debug message", console_output.getvalue())
		self.assertIn("info message", console_output.getvalue())

	def test_writes_debug_to_file_and_info_to_console(self):
		with tempfile.TemporaryDirectory() as temp_dir:
			log_path = Path(temp_dir) / "nested" / "training.log"
			console_output = io.StringIO()

			with patch("sys.stderr", console_output):
				logger = configure_logger(str(log_path))
				logger.debug("file-only debug message")
				logger.info("shared info message")

			log_contents = log_path.read_text()

		self.assertIn("file-only debug message", log_contents)
		self.assertIn("shared info message", log_contents)
		self.assertNotIn("file-only debug message", console_output.getvalue())
		self.assertIn("shared info message", console_output.getvalue())
		self.assertEqual(len(logger.handlers), 2)
		self.assertEqual(logger.handlers[1].level, logging.DEBUG)


if __name__ == "__main__":
	unittest.main()
