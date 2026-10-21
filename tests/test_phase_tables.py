import argparse
import tempfile
import unittest
from pathlib import Path
from preprocess.preprocess import PHASE_TABLES, resolve_phase_table


class PhaseTableTests(unittest.TestCase):
    def test_split_defaults_and_override_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for filename in PHASE_TABLES.values():
                (root / filename).write_text('{}')
            args = argparse.Namespace(data_root=d, sax_phase_info=None)
            for split, filename in PHASE_TABLES.items():
                self.assertEqual(resolve_phase_table(args, split), root / filename)
            common = root / 'common.json'; common.write_text('{}')
            specific = root / 'specific.json'; specific.write_text('{}')
            args.sax_phase_info = str(common)
            args.sax_phase_info_val = str(specific)
            self.assertEqual(resolve_phase_table(args, 'train'), common)
            self.assertEqual(resolve_phase_table(args, 'val'), specific)
            specific.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'val'):
                resolve_phase_table(args, 'val')

    def test_no_fallback_from_validation_to_training_table(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / PHASE_TABLES['train']).write_text('{}')
            args = argparse.Namespace(data_root=d, sax_phase_info=None)
            with self.assertRaisesRegex(FileNotFoundError, 'id_slice_info_valid.json'):
                resolve_phase_table(args, 'val')
