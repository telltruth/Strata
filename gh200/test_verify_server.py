import unittest
from verify_server import check

class VerifyServerTest(unittest.TestCase):
    def test_confirm_allocation(self):
        s = {"loaded": True, "context": {"max_positions": 262144},
             "concurrency": {"serving": 8}, "model": "strata"}
        value = check(s)
        self.assertEqual(value["allocated_slots"], 8)
        self.assertEqual(value["verified_context_ceiling"], 262144)

    def test_not_silently_accept_fewer_slots(self):
        with self.assertRaisesRegex(ValueError, "allocated 4 slots"):
            check({"loaded": True, "context": {"max_positions": 262144},
                   "concurrency": {"serving": 4}})

    def test_not_silently_accept_shorter_context(self):
        with self.assertRaisesRegex(ValueError, "context only"):
            check({"loaded": True, "context": {"max_positions": 65536},
                   "concurrency": {"serving": 8}})

    def test_not_loaded(self):
        with self.assertRaisesRegex(ValueError, "not loaded"):
            check({"loaded": False})

if __name__ == "__main__":
    unittest.main()
