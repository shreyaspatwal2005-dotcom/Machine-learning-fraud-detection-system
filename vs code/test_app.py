"""API tests for app.py (standard library unittest + Flask test client).
Run:  python -m unittest -v test_app
"""
import io
import unittest

import app as A
from generate_data import generate

RISKY = dict(amount=1200, hour=2, distance_from_home=300, txn_last_24h=7, account_age_days=40, avg_amount_30d=45, amount_ratio=26.7)
LEGIT = dict(amount=25, hour=14, distance_from_home=3, txn_last_24h=1, account_age_days=2500, avg_amount_30d=40, amount_ratio=0.62)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = A.app.test_client()
        cls.sample = cls.c.post("/api/sample").get_json()
        cls.trained = cls.c.post("/api/train").get_json()

    def test_01_index_page(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Fraud Desk", r.data)
        r.close()

    def test_02_sample_report(self):
        rep = self.sample["report"]
        self.assertTrue(self.sample["ok"])
        self.assertEqual(rep["rows"], 20000)
        self.assertEqual(rep["n_fraud"] + rep["n_legit"], rep["rows"])
        self.assertNotIn("is_fraud", rep["features"])

    def test_03_train(self):
        t = self.trained
        self.assertTrue(t["ok"])
        self.assertEqual(t["n_train"] + t["n_test"], 20000)
        self.assertNotIn("is_fraud", t["features"])
        self.assertEqual(set(t["profiles"]), {"0", "1"})

    def test_04_results(self):
        r = self.c.get("/api/results?model=rf&held=lr&threshold=0.5").get_json()
        self.assertEqual([x["key"] for x in r["rows"]], ["base", "lr", "rf"])
        base = r["rows"][0]
        self.assertEqual(base["recall"], 0)           # baseline catches no fraud...
        self.assertGreater(base["accuracy"], 0.97)    # ...yet looks accurate
        self.assertEqual(len(r["curve"]["t"]), 91)
        self.assertEqual(sum(r["cm"].values()), r["n_test"])
        self.assertEqual(sum(g["total"] for g in r["held"]), r["n_test"])
        for g in r["held"]:
            for row in g["rows"]:
                self.assertEqual(row["actual"] == row["predicted"], g["good"])

    def test_05_threshold_changes_results(self):
        lo = self.c.get("/api/results?threshold=0.1").get_json()["rows"][2]["recall"]
        hi = self.c.get("/api/results?threshold=0.9").get_json()["rows"][2]["recall"]
        self.assertGreaterEqual(lo, hi)

    def test_06_predict(self):
        r = self.c.post("/api/predict", json={"values": RISKY, "model": "rf"}).get_json()
        self.assertEqual(r["prediction"], "FRAUD")
        self.assertTrue(0 <= r["fraud_probability"] <= 1)
        r = self.c.post("/api/predict", json={"values": LEGIT, "model": "lr"}).get_json()
        self.assertEqual(r["prediction"], "LEGITIMATE")

    def test_07_predict_empty_field_and_errors(self):
        ok = self.c.post("/api/predict", json={"values": {**RISKY, "amount": ""}})
        self.assertEqual(ok.get_json()["empty"], 1)
        bad = self.c.post("/api/predict", json={"values": {**RISKY, "amount": "abc"}})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("Not a number", bad.get_json()["error"])
        self.assertEqual(self.c.post("/api/predict", json={"values": {**LEGIT, "is_fraud": 1}}).status_code, 400)
        self.assertEqual(self.c.post("/api/predict", json={"values": {"amount": 1}}).status_code, 400)
        self.assertEqual(self.c.post("/api/predict", json={"values": LEGIT, "model": "x"}).status_code, 400)

    def test_08_test_cases_all_pass(self):
        self.assertTrue(self.c.get("/api/testcases").get_json()["ok"])
        r = self.c.post("/api/testcases/run", json={"model": "rf", "threshold": 0.5}).get_json()
        self.assertEqual(len(r["pred"]), 8)
        self.assertTrue(all(p["passed"] for p in r["pred"]))
        self.assertEqual(len(r["val"]), 9)
        self.assertTrue(all(v["passed"] for v in r["val"]))

    def test_09_needs_training_and_upload_validation(self):  # runs last: resets the shared state
        bad = self.c.post("/api/upload", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")}, content_type="multipart/form-data")
        self.assertFalse(bad.get_json()["ok"])
        self.assertIn("is_fraud", bad.get_json()["error"])
        self.assertEqual(self.c.get("/api/results").status_code, 409)
        self.assertEqual(self.c.post("/api/train").status_code, 400)
        csv = generate(1500, seed=3).to_csv(index=False).encode()
        up = self.c.post("/api/upload", data={"file": (io.BytesIO(csv), "t.csv")}, content_type="multipart/form-data").get_json()
        self.assertTrue(up["ok"])
        self.assertTrue(self.c.post("/api/train").get_json()["ok"])
        other = generate(1500, seed=4).rename(columns={"amount": "value"}).to_csv(index=False).encode()
        self.c.post("/api/upload", data={"file": (io.BytesIO(other), "o.csv")}, content_type="multipart/form-data")
        self.c.post("/api/train")
        self.assertFalse(self.c.get("/api/testcases").get_json()["ok"])  # sample-only cases are skipped


if __name__ == "__main__":
    unittest.main(verbosity=2)
