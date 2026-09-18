"""CPU-only tests; outside tests/ to preserve the running predecessor's file set."""
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from run_cls_aux_queue import REPO,NAMES,WEIGHTS,validate_config,assert_completed,verify_hashes,digest


class QueueTests(unittest.TestCase):
    def test_configs_only_change_aux(self):
        base=yaml.safe_load((REPO/'config/labeling_coarse_aux.yaml').read_text())
        for name,weight in zip(NAMES,WEIGHTS):
            cfg=yaml.safe_load((REPO/'config/cls_aux_queue'/f'{name}.yaml').read_text())
            validate_config(base,cfg,name,weight)
            cfg['encoder_lr']=.1
            with self.assertRaises(ValueError):validate_config(base,cfg,name,weight)

    def test_predecessor_failure_stops_queue(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(RuntimeError):assert_completed(root,'abc')
            path=root/'training_complete.json'
            path.write_text(json.dumps(dict(status='completed',manifest_sha256='wrong')))
            with self.assertRaises(RuntimeError):assert_completed(root,'abc')
            path.write_text(json.dumps(dict(status='completed',manifest_sha256='abc')))
            assert_completed(root,'abc')

    def test_pins_detect_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'input.txt';p.write_text('original')
            pins={'input.txt':digest(p)};verify_hashes(root,pins)
            p.write_text('changed')
            with self.assertRaises(RuntimeError):verify_hashes(root,pins)


if __name__=='__main__':unittest.main()
