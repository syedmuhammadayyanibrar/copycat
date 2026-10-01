import unittest
import numpy as np
from copycat.data.generator import SyntheticDatasetGenerator
from copycat.data.schema import FORMATS, THUMBNAIL_STYLES

class TestSyntheticDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SyntheticDatasetGenerator(seed=123)

    def test_creator_profile_generation(self):
        profile = self.generator.generate_creator_profile("creator_test")
        self.assertEqual(profile.creator_id, "creator_test")
        self.assertEqual(len(profile.taste_vector), 5)
        self.assertAlmostEqual(float(np.linalg.norm(profile.taste_vector)), 1.0, places=5)
        self.assertEqual(len(profile.familiar_formats), 2)
        self.assertEqual(len(profile.familiar_thumbnails), 1)

    def test_video_record_splits(self):
        profile = self.generator.generate_creator_profile("creator_test")
        video_sim = self.generator.generate_video(profile, "v_sim", "held_out_similar")
        video_nov = self.generator.generate_video(profile, "v_nov", "held_out_novel")
        
        self.assertIn(video_sim.format, profile.familiar_formats)
        self.assertEqual(video_sim.novelty_dist, 0.0)
        self.assertNotIn(video_nov.format, profile.familiar_formats)
        self.assertEqual(video_nov.novelty_dist, 1.0)
        self.assertTrue(video_sim.performance > 0.0)
        self.assertTrue(video_nov.performance > 0.0)

    def test_bundle_generation(self):
        bundle = self.generator.generate_bundle(n_creators=3, n_train=10, n_similar=5, n_novel=5)
        self.assertEqual(len(bundle.creators), 3)
        self.assertEqual(len(bundle.train_df), 30)
        self.assertEqual(len(bundle.similar_df), 15)
        self.assertEqual(len(bundle.novel_df), 15)

if __name__ == "__main__":
    unittest.main()
