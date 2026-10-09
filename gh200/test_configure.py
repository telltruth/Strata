import unittest
from configure import configure, set_argument

class ConfigureTests(unittest.TestCase):
    def test_profile_and_original(self):
        original = {"args": ["--pack", "/data/p", "--native", "/data/m.gguf",
                             "--max-context", "32768", "--expert-cache", "500"]}
        prior = list(original["args"])
        config = configure(original)
        self.assertEqual(original["args"], prior)
        self.assertEqual(config["parallel"], 8)
        self.assertEqual(config["args"].count("--max-context"), 1)
        self.assertEqual(config["args"][config["args"].index("--max-context")+1], "262144")
        self.assertEqual(config["args"][config["args"].index("--kv-resident")+1], "32768")
        self.assertEqual(config["args"][config["args"].index("--kv")+1], "int8")
        self.assertEqual(config["args"][config["args"].index("--expert-cache")+1], "auto")
        self.assertFalse(config["reasoning_loop_recovery"])
        self.assertEqual(config["reasoning_budget_tokens"], 0)

    def test_reapply(self):
        base = {"args": ["--pack", "/p", "--native", "/n", "--batch", "4"]}
        once = configure(base)
        self.assertEqual(once, configure(once))

    def test_invalid_kv(self):
        base = {"args": ["--pack", "/p", "--native", "/n", "--kv", "k8v4"]}
        with self.assertRaisesRegex(ValueError, "k8v4"):
            configure(base)

    def test_invalid_count(self):
        base = {"args": ["--pack", "/p", "--native", "/n"]}
        with self.assertRaisesRegex(ValueError, "slots"):
            configure(base, slots=9)

    def test_missing_install(self):
        with self.assertRaisesRegex(ValueError, "install a model"):
            configure({"args": []})

    def test_missing_arg_value(self):
        with self.assertRaisesRegex(ValueError, "missing its value"):
            set_argument(["--max-context"], "--max-context", 262144)

if __name__ == "__main__":
    unittest.main()
