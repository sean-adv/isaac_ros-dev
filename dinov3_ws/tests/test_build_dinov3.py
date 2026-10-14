"""CPU checks for bundle extraction and accuracy evidence handling."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('build_dinov3', Path(__file__).parents[1] / 'build_dinov3.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BuildDinoTests(unittest.TestCase):
    def test_extract_uses_section_offset_and_length(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = {'sections': {'engine.plan': {'offset': 4, 'length': 6}}}
            encoded = json.dumps(header).encode()
            bundle = root / 'model.bundle'
            bundle.write_bytes(b'BUNDLE\x01\x00' + struct.pack('<Q', len(encoded)) + encoded + b'skipENGINEtrailing')
            engine = root / 'model.engine'
            self.assertEqual(builder.extract(bundle, engine), header)
            self.assertEqual(engine.read_bytes(), b'ENGINE')
            bundle.write_bytes(bundle.read_bytes()[:-10])
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                builder.extract(bundle, engine)

    def test_scale_error_is_visible_even_when_cosine_is_perfect(self):
        hidden = np.random.default_rng(0).normal(size=(1, 201, 384))
        expected = dict(last_hidden_state=hidden, pooler_output=hidden[:, 0])
        actual = {k: v * 1.1 for k, v in expected.items()}
        result = builder.metrics(actual, expected, registers=4)
        self.assertGreater(result['full_cosine'], .999999)
        self.assertGreater(result['relative_frobenius'], builder.THRESHOLDS['relative_frobenius'])

    def test_nonfinite_output_is_rejected(self):
        hidden = np.ones((1, 201, 384))
        expected = dict(last_hidden_state=hidden, pooler_output=hidden[:, 0])
        actual = {k: v.copy() for k, v in expected.items()}
        actual['pooler_output'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            builder.metrics(actual, expected, registers=4)


if __name__ == '__main__':
    unittest.main()
