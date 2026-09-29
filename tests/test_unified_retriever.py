import time
import unittest

from vector_db.unified_retriever import UnifiedRetriever


class UnifiedRetrieverTests(unittest.TestCase):
    def make_retriever(self):
        retriever = UnifiedRetriever.__new__(UnifiedRetriever)
        retriever.vector_manager = object()
        retriever._run_ner = lambda query: {"疾病": "感冒"}
        retriever._run_intent = lambda query: '["症状"]'
        retriever._detect_patient_record_query = lambda query: (False, "")
        return retriever

    def test_cross_source_rrf_is_deterministic(self):
        retriever = self.make_retriever()
        merged = retriever._merge_results(
            [{"id": "kg-1", "content": "图谱证据"}],
            [{"id": "vec-1", "content": "向量证据", "score": 0.9}],
        )
        self.assertEqual([row["id"] for row in merged], ["kg-1", "vec-1"])
        self.assertAlmostEqual(merged[0]["rrf_score"], 1 / 61)
        self.assertAlmostEqual(merged[1]["rrf_score"], 1 / 61)

    def test_public_kg_and_vector_branches_run_in_parallel(self):
        retriever = self.make_retriever()

        def slow_kg(entities, intents):
            time.sleep(0.15)
            return [{"id": "kg-1", "content": "图谱证据"}]

        def slow_vector(*args, **kwargs):
            time.sleep(0.15)
            return {"results": [{"id": "vec-1", "content": "向量证据"}]}

        retriever._search_kg = slow_kg
        retriever._search_vector = slow_vector
        started = time.perf_counter()
        result = retriever.retrieve("感冒有什么症状", use_rerank=False, include_prompt=False)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.27)
        self.assertEqual(len(result["merged_results"]), 2)

    def test_patient_graph_requires_authenticated_owner(self):
        retriever = self.make_retriever()
        self.assertEqual(retriever._search_patient_kg("张三", user_id=None), [])


if __name__ == "__main__":
    unittest.main()
