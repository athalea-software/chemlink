import io
import logging
import subprocess
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from pipelines.dynamics.logger import get_step_logger, warnings_to_log
from utils.retry import SubprocessError, run_subprocess


class WarningLogTests(unittest.TestCase):
    def test_diagnostics_saved_without_console_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'simulation.log'
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                logger = get_step_logger('test.warning_log', str(path))
                original = warnings.showwarning
                try:
                    with self.assertRaisesRegex(ValueError, 'repair failed'):
                        with warnings_to_log(logger):
                            warnings.warn('duplicate atom 1HH', UserWarning)
                            raise ValueError('repair failed')
                    self.assertIs(warnings.showwarning, original)
                    detail = 'x' * 500 + 'missing atom at end'
                    with patch('utils.retry.subprocess.run', side_effect=subprocess.CalledProcessError(
                        1, ['gmx'], stderr=detail,
                    )):
                        with self.assertRaises(SubprocessError):
                            run_subprocess(['gmx'], logger=logger, file_only=True)
                    self.assertEqual(stdout.getvalue(), '')
                    self.assertEqual(stderr.getvalue(), '')
                    contents = path.read_text()
                    self.assertIn('UserWarning: duplicate atom 1HH', contents)
                    self.assertIn(detail, contents)
                    logger.error('terminal failure')
                    self.assertIn('terminal failure', stdout.getvalue())
                finally:
                    for handler in list(logger.handlers):
                        handler.close()
                        logger.removeHandler(handler)
