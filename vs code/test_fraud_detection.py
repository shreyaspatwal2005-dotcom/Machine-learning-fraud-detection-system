"""Automated tests (standard library only). Run with:
    python -m unittest -v test_fraud_detection
"""
import unittest

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score

import fraud_detection as fd
from generate_data import generate

GOOD = dict(amount=25, hour=14, distance_from_home=3, txn_last_24h=1,
            account_age_days=2500, avg_amount_30d=40, amount_ratio=0.62)
RISKY = dict(amount=1200, hour=2, distance_from_home=300, txn_last_24h=7,
             account_age_days=40, avg_amount_30d=45, amount_ratio=26.7)


class FraudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fd.validate_data(generate(n=6000, seed=1))
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = fd.split_data(cls.df)
        model = fd.build_model("random_forest").fit(cls.X_train, cls.y_train)
        cls.bundle = {"model": model, "features": list(cls.X_train.columns)}

    # ---- Requirement 2: validation ----
    def test_01_missing_target_column(self):
        with self.assertRaises(ValueError):
            fd.validate_data(self.df.drop(columns="is_fraud"))

    def test_02_invalid_labels(self):
        with self.assertRaises(ValueError):
            fd.validate_data(self.df.assign(is_fraud=self.df.is_fraud.replace({1: 2})))

    def test_03_nan_labels(self):
        bad = self.df.assign(is_fraud=self.df.is_fraud.astype(float).where(self.df.index > 5))
        with self.assertRaises(ValueError):
            fd.validate_data(bad)

    def test_04_single_class(self):
        with self.assertRaises(ValueError):
            fd.validate_data(self.df.assign(is_fraud=0))

    def test_05_non_numeric_feature(self):
        with self.assertRaises(ValueError):
            fd.validate_data(self.df.assign(merchant="shop"))

    # ---- Requirement 3: stratified split ----
    def test_06_split_is_stratified(self):
        self.assertLess(abs(self.y_train.mean() - self.y_test.mean()), 0.003)
        self.assertEqual(len(self.X_train) + len(self.X_test), len(self.df))

    def test_07_target_not_in_features(self):
        self.assertNotIn("is_fraud", self.X_train.columns)

    # ---- Requirements 4-6: model vs baseline ----
    def test_08_model_beats_baseline_recall(self):
        base = DummyClassifier(strategy="most_frequent").fit(self.X_train, self.y_train)
        lr = fd.build_model("logistic").fit(self.X_train, self.y_train)
        self.assertEqual(recall_score(self.y_test, base.predict(self.X_test)), 0)
        self.assertGreater(recall_score(self.y_test, lr.predict(self.X_test)), 0.5)

    # ---- Requirement 7: prediction ----
    def test_09_predict_legitimate(self):
        r = fd.predict_transaction(self.bundle, GOOD)
        self.assertEqual(r["prediction"], "LEGITIMATE")
        self.assertLess(r["fraud_probability"], 0.5)

    def test_10_predict_fraud(self):
        r = fd.predict_transaction(self.bundle, RISKY)
        self.assertEqual(r["prediction"], "FRAUD")
        self.assertGreaterEqual(r["fraud_probability"], 0.5)

    def test_11_probability_in_range(self):
        p = fd.predict_transaction(self.bundle, RISKY)["fraud_probability"]
        self.assertTrue(0 <= p <= 1)

    def test_12_predict_handles_missing_value(self):
        r = fd.predict_transaction(self.bundle, {**RISKY, "amount": np.nan})
        self.assertIn(r["prediction"], ("FRAUD", "LEGITIMATE"))

    def test_13_predict_rejects_target(self):
        with self.assertRaises(ValueError):
            fd.predict_transaction(self.bundle, {**GOOD, "is_fraud": 1})

    def test_14_predict_rejects_missing_features(self):
        with self.assertRaises(ValueError):
            fd.predict_transaction(self.bundle, {"amount": 5})

    def test_15_threshold_changes_decision(self):
        self.assertEqual(fd.predict_transaction(self.bundle, GOOD, threshold=0.0)["prediction"], "FRAUD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
