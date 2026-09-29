import unittest

from vector_db.vector_manager import VectorManager


class VectorTenantBoundaryTests(unittest.TestCase):
    def make_manager(self):
        manager = VectorManager.__new__(VectorManager)
        manager.fallback_triggered = False
        manager.last_search_mode = "dense"
        return manager

    def test_personal_collection_rejects_missing_user_id(self):
        manager = self.make_manager()
        manager.search_dense = lambda *args, **kwargs: self.fail("search must not run")
        result = manager.search("血压", collection_type="scenario_memory", user_id=None)
        self.assertFalse(result["success"])
        self.assertIn("拒绝检索个人数据", result["message"])

    def test_dense_degradation_preserves_user_and_report_filters(self):
        manager = self.make_manager()
        calls = []

        def dense(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("primary failure")
            return []

        manager.search_dense = dense
        result = manager.search(
            "我的血压", search_type="dense", collection_type="scenario_memory",
            user_id="alice", report_only=True,
        )
        self.assertFalse(result["success"])
        self.assertEqual(calls[-1]["user_filter"], "alice")
        self.assertTrue(calls[-1]["report_only"])


if __name__ == "__main__":
    unittest.main()
