import unittest

from goose_semseg.data.spectral_tiles import aligned_tile_positions as aligned_positions


class OverlapEvaluationTests(unittest.TestCase):
    def test_positions_cover_without_padding_and_align_far_edge(self):
        for length in (1503, 2055, 5460, 8192):
            positions = aligned_positions(length, 768, 512)
            self.assertEqual(positions[0], 0)
            self.assertEqual(positions[-1], length - 768)
            self.assertEqual(positions, sorted(set(positions)))
            coverage = [False] * length
            for start in positions:
                self.assertGreaterEqual(start, 0)
                self.assertLessEqual(start + 768, length)
                coverage[start:start + 768] = [True] * 768
            self.assertTrue(all(coverage))

    def test_rejects_axis_smaller_than_tile(self):
        with self.assertRaises(ValueError):
            aligned_positions(767, 768, 512)


if __name__ == '__main__':
    unittest.main()
