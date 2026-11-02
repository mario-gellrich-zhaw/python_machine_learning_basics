import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from enron_data import load_email_splits


class PreparedEmailTextTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'emails.csv.gz'
        messages = [
            'Team\u00a0meeting\tat  noon',
            'Invoice\u2003review\n  today',
            'Gift\vnotice\r\n now',
        ]
        normalized = [
            'team\u00a0meeting at noon',
            'invoice\u2003review today',
            'gift\vnotice now',
        ]
        self.frame = pd.DataFrame({
            'message': messages,
            'label': [0, 1, 0],
            'split': ['train', 'validation', 'test'],
            'source_row': [12, 4, 9],
            'message_key': [
                hashlib.sha256(text.encode('utf-8')).hexdigest()
                for text in normalized
            ],
        })

    def write_data(self):
        self.frame.to_csv(self.path, index=False, compression='gzip')

    def check_round_trip(self):
        self.write_data()
        splits = load_email_splits(self.path, seed=123)
        for index, split in enumerate(splits):
            self.assertEqual(split['body'].tolist(), [self.frame.loc[index, 'message']])
            self.assertEqual(split['message_key'].tolist(), [self.frame.loc[index, 'message_key']])
            self.assertEqual(split['source_row'].tolist(), [self.frame.loc[index, 'source_row']])
            self.assertEqual(split['label'].tolist(), [self.frame.loc[index, 'label']])

    def test_prepared_whitespace_with_python_strings(self):
        with pd.option_context('mode.string_storage', 'python'):
            self.check_round_trip()

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'Optional Arrow backend is not installed.')
    def test_prepared_whitespace_with_arrow_strings(self):
        with pd.option_context('mode.string_storage', 'pyarrow'):
            self.check_round_trip()

    def test_tampered_content_is_rejected(self):
        self.frame.loc[0, 'message'] = 'Changed meeting details'
        self.write_data()
        with self.assertRaisesRegex(ValueError, 'mismatched message content'):
            load_email_splits(self.path)

    def test_duplicate_content_is_rejected(self):
        self.frame.loc[1, ['message', 'message_key']] = self.frame.loc[
            0, ['message', 'message_key']].to_numpy()
        self.write_data()
        with self.assertRaisesRegex(ValueError, 'duplicate or mismatched message content'):
            load_email_splits(self.path)

    def test_empty_content_is_rejected(self):
        self.frame.loc[0, 'message'] = ' \t\n'
        self.write_data()
        with self.assertRaisesRegex(ValueError, 'empty, duplicate or mismatched message content'):
            load_email_splits(self.path)


if __name__ == '__main__':
    unittest.main()
