import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goose_semseg.data.spectral_tiles import resolve_manifest_source


class PortablePathTests(unittest.TestCase):
    def test_environment_override_moves_absolute_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'repo' / 'data' / 'prepared'
            source = Path(temporary) / 'copied_raw'
            root.mkdir(parents=True)
            source.mkdir()
            with patch.dict(os.environ, {'RIVER_SEMSEG_SOURCE_ROOT': str(source)}):
                resolved = resolve_manifest_source(root, '/old/server/Labeling_Data_v2')
            self.assertEqual(resolved, source.resolve())

    def test_repository_basename_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / 'repo'
            root = repo / 'data' / 'prepared'
            source = repo / 'Labeling_Data_v2'
            root.mkdir(parents=True)
            source.mkdir()
            with patch.dict(os.environ, {}, clear=True):
                resolved = resolve_manifest_source(root, '/old/server/Labeling_Data_v2')
            self.assertEqual(resolved, source.resolve())


if __name__ == '__main__':
    unittest.main()
