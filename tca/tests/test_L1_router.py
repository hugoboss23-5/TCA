"""Tests for TCA Layer 1: Chestohedron Router."""

import unittest

from tca.L1_router.router import route, GATES


class TestL1Router(unittest.TestCase):

    def test_factual_query_mirror_and_express_dominate(self):
        """Simple factual query -> MIRROR and EXPRESS should be among the top gates."""
        weights = route("What is reservoir computing? Explain how it works.")
        # MIRROR and EXPRESS should be the two highest gates.
        sorted_gates = sorted(GATES, key=lambda g: weights[g], reverse=True)
        top_two = set(sorted_gates[:2])
        self.assertIn("MIRROR", top_two,
                       f"MIRROR should be in top 2, got {sorted_gates[:3]}")
        self.assertIn("EXPRESS", top_two,
                       f"EXPRESS should be in top 2, got {sorted_gates[:3]}")

    def test_creative_query_darash_high(self):
        """Creative/exploratory query -> DARASH should be highest."""
        weights = route("What if we could explore new possibilities and imagine "
                        "a speculative hypothesis about unknown phenomena?")
        max_gate = max(GATES, key=lambda g: weights[g])
        self.assertEqual(max_gate, "DARASH",
                         f"DARASH should dominate, but {max_gate} was highest")

    def test_self_referential_verify_and_remove_dominate(self):
        """Self-referential query -> VERIFY and REMOVE should dominate."""
        weights = route("Am I reasoning correctly or is something wrong and invalid?")
        sorted_gates = sorted(GATES, key=lambda g: weights[g], reverse=True)
        top_two = set(sorted_gates[:2])
        self.assertIn("VERIFY", top_two,
                       f"VERIFY should be in top 2, got {sorted_gates[:3]}")
        self.assertIn("REMOVE", top_two,
                       f"REMOVE should be in top 2, got {sorted_gates[:3]}")

    def test_all_gates_nonzero(self):
        """All 7 gates return nonzero values for any input."""
        test_inputs = [
            "hello",
            "What is the meaning of life?",
            "",
            "12345",
            "The quick brown fox jumps over the lazy dog.",
        ]
        for text in test_inputs:
            weights = route(text)
            for gate in GATES:
                self.assertGreater(weights[gate], 0,
                                   f"Gate {gate} was zero for input {text!r}")

    def test_weights_normalized(self):
        """Gate weights sum to 1.0 for any input."""
        test_inputs = [
            "What is this?",
            "Explore the unknown possibilities of imagined worlds.",
            "Is this correct? Prove it.",
            "This is wrong and contradicts everything.",
            "",
        ]
        for text in test_inputs:
            weights = route(text)
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=6,
                                   msg=f"Weights sum to {total} for {text!r}")


if __name__ == "__main__":
    unittest.main()
