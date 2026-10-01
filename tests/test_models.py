import unittest
import numpy as np
from copycat.data.generator import SyntheticDatasetGenerator
from copycat.models.memorization import MemorizationBaseline
from copycat.models.taste import LatentTasteModel
from copycat.models.oracle import OracleModel

class TestModels(unittest.TestCase):
    def setUp(self):
        generator = SyntheticDatasetGenerator(seed=101)
        self.bundle = generator.generate_bundle(n_creators=3, n_train=20, n_similar=10, n_novel=10)

    def test_memorization_baseline(self):
        model = MemorizationBaseline(top_k=3)
        model.fit(self.bundle.train_df)
        preds_sim = model.predict(self.bundle.similar_df)
        preds_nov = model.predict(self.bundle.novel_df)
        
        self.assertEqual(len(preds_sim), len(self.bundle.similar_df))
        self.assertEqual(len(preds_nov), len(self.bundle.novel_df))
        self.assertTrue(np.all(preds_sim > 0.0))
        self.assertTrue(np.all(preds_nov > 0.0))

    def test_latent_taste_model(self):
        model = LatentTasteModel(k_factors=5, epochs=30)
        model.fit(self.bundle.train_df)
        
        factors = model.get_latent_factors(self.bundle.similar_df)
        self.assertEqual(factors.shape, (len(self.bundle.similar_df), 5))
        
        preds_sim = model.predict(self.bundle.similar_df)
        preds_nov = model.predict(self.bundle.novel_df)
        self.assertEqual(len(preds_sim), len(self.bundle.similar_df))
        self.assertEqual(len(preds_nov), len(self.bundle.novel_df))
        self.assertTrue(np.all(preds_sim > 0.0))
        self.assertTrue(np.all(preds_nov > 0.0))

    def test_oracle_model(self):
        oracle = OracleModel(creators=self.bundle.creators)
        oracle.fit(self.bundle.train_df)
        preds = oracle.predict(self.bundle.similar_df)
        self.assertEqual(len(preds), len(self.bundle.similar_df))
        self.assertTrue(np.all(preds > 0.0))

if __name__ == "__main__":
    unittest.main()
