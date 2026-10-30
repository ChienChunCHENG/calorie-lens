import json
import math
import os
import unittest

import range_engine as R

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "spec", "vectors.json")


def run(c):
    i = c["input"]
    if c["fn"] == "estimateSolid":
        return R.estimate_solid(i["dish"], i["volumeMl"])
    if c["fn"] == "estimateDrink":
        return R.estimate_drink(i["drink"], i["sizeMl"], i["sugarLevel"],
                                i["fullSugarGPer700"], i["sugarFraction"])
    return R.combine_meal(i["items"])


class Vectors(unittest.TestCase):
    def test_vectors(self):
        with open(PATH, encoding="utf-8") as f:
            cases = json.load(f)["cases"]
        self.assertGreaterEqual(len(cases), 20)
        for c in cases:
            with self.subTest(c["name"]):
                e = c["expected"]
                if e.get("error"):
                    with self.assertRaises(ValueError):
                        run(c)
                    continue
                r = run(c)
                for k in ("lo", "med", "hi"):
                    self.assertTrue(math.isclose(r[k], e[k], rel_tol=1e-9, abs_tol=1e-12), k)
                self.assertEqual(r.get("flags", []), e["flags"])

    def test_format(self):
        self.assertEqual(R.format_range(447.4, 602.6), "445–605")


if __name__ == "__main__":
    unittest.main()
