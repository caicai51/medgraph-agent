import unittest

from scripts.retrieval_diagnostics import ndcg_at_k, stage_metrics, unique_rows


def row(relevant, retrieved):
    return {"relevant_doc_ids": relevant, "stages": {"final_top10": {"ids": retrieved}}}


class RetrievalMetricTests(unittest.TestCase):
    def test_hand_calculated_macro_metrics_and_mrr_at_10(self):
        rows = [row(["a"], ["a"]), row(["b", "c"], ["c", "x"]), row(["d"], [])]
        metrics = stage_metrics(rows, "final_top10")
        self.assertAlmostEqual(metrics["Recall@10"], (1 + 0.5 + 0) / 3)
        self.assertAlmostEqual(metrics["HitRate@10"], 2 / 3)
        self.assertAlmostEqual(metrics["MRR@10"], (1 + 1 + 0) / 3)
        self.assertAlmostEqual(metrics["nDCG@10"], (1 + (1 / (1 + 1 / 1.5849625007)) + 0) / 3, places=6)

    def test_pure_reranking_same_top10_keeps_recall_and_hit_rate(self):
        before = [row(["a", "b"], ["a", "x", "b"])]
        after = [row(["a", "b"], ["b", "x", "a"])]
        self.assertEqual(stage_metrics(before, "final_top10")["Recall@10"], stage_metrics(after, "final_top10")["Recall@10"])
        self.assertEqual(stage_metrics(before, "final_top10")["HitRate@10"], stage_metrics(after, "final_top10")["HitRate@10"])

    def test_ndcg_is_bounded(self):
        self.assertEqual(ndcg_at_k(["a", "b"], {"a", "b"}), 1.0)
        self.assertEqual(ndcg_at_k([], {"a"}), 0.0)

    def test_duplicate_ids_do_not_get_counted_twice(self):
        duplicate_results = [row(["a", "b"], ["a", "a", "x"])]
        values = stage_metrics(duplicate_results, "final_top10")
        self.assertAlmostEqual(values["Recall@10"], 0.5)
        self.assertAlmostEqual(values["HitRate@10"], 1.0)

    def test_rerank_output_must_be_drawn_from_candidates(self):
        candidates = [{"id": "a"}, {"id": "b"}, {"id": "b"}]
        output = [{"id": "b"}, {"id": "a"}]
        candidate_ids = {item["id"] for item in unique_rows(candidates)}
        self.assertTrue({item["id"] for item in output}.issubset(candidate_ids))


if __name__ == "__main__":
    unittest.main()
