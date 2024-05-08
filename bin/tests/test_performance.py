import unittest
from bin.src.utils.performance import Performance

class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.labels = [0, 1, 0, 1]
        self.predictions = [0.1, 0.9, 0.7, 0.6]

    def test_metrics(self):
        target = {
            "rocauc": 0.75,
            "prauc": 0.83,
            "mcc": 0.58,
            "f1score": 0.8,
            "precision": 0.67,
            "recall": 1.0,
            "spearmanr": 0.45
        }
        for metric, val in target.items():
            perf = round(Performance(self.labels, self.predictions, metric=metric).val, 2)
            self.assertEqual(perf, val)
