import unittest

from tools.transport.score_deadlines import score


class CaptureScores(unittest.TestCase):
    def test_clock_error_cannot_make_an_update_arrive_before_submission(self):
        row = dict(session="a", source="s", attempt="1", receiver="0", bytes=32,
                   generated_ns=0, tx_ns=10, rx_ns=9)
        result = score([row], deadline=8, error=2)
        self.assertEqual(result["first_by_generation_deadline"], [0, 0])
        self.assertEqual(result["per_link"][0]["by_generation_deadline"], [0, 0])

    def test_generation_deadlines_quorum_and_out_of_order_freshness(self):
        rows = []
        for attempt, (tx, generation, receives) in enumerate((
                (10, 0, (35, 15, None)), (20, 18, (25, None, 29)), (40, 40, (None, None, None)))):
            for receiver, arrival in enumerate(receives):
                rows.append(dict(session="a", source="s", attempt=str(attempt), receiver=str(receiver),
                                 bytes="32", tx_ns=str(tx), generated_ns=str(generation),
                                 rx_ns="" if arrival is None else str(arrival)))
        result = score(reversed(rows), deadline=12, error=1)
        self.assertEqual(result["receiver_opportunities"], 9)
        self.assertEqual(result["first_by_generation_deadline"], [1/3,1/3])
        self.assertEqual(result["quorum_by_generation_deadline"], [1/3,1/3])
        self.assertEqual(result["all_by_generation_deadline"], [0,0])
        first = result["per_link"][0]["age_before_offer_latest_rx"]
        self.assertEqual((first["known"], first["unknown"]), (1,2))
        self.assertEqual(first["quantiles_ns"]["0.95"], 22)
        with self.assertRaisesRegex(ValueError, "exceeds eligible"):
            score(rows, 12, 1, quorum=4)
