import unittest

from tools.transport.heartbeat import compare, recovery_by_age


class HeartbeatValidation(unittest.TestCase):
    def test_paired_schedule_gaps_support_and_failure(self):
        measured = {("sensor", i): 0 for i in (0, 1, 2, 4, 5)}
        simulated = dict(measured)
        simulated["sensor", 1] = 1
        self.assertEqual(compare(measured, simulated, 1, .05)["sensor"]["verdict"], "rejected")
        self.assertEqual(compare(measured, measured, 2, .05)["sensor"]["verdict"], "insufficient_evidence")
        self.assertEqual(compare(measured, measured, 1, 0)["sensor"]["verdict"], "qualified")
        del simulated["sensor", 2]
        with self.assertRaises(ValueError):
            compare(measured, simulated, 1, .05)

    def test_recovery_excludes_unknown_onsets_and_unobserved_successors(self):
        result = recovery_by_age([[0, 0, 7, 0, 0, 7, 0], [7, 0, 7]])
        self.assertEqual(result["after_one"], dict(recoveries=1, successors=2))
        self.assertEqual(result["after_two_or_more"], dict(recoveries=1, successors=1))
