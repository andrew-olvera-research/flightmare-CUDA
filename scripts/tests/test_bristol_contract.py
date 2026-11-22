import tempfile
import unittest
from pathlib import Path

from tools.transport.validate_bristol import broadcast_rows, read


class RoadsideCapture(unittest.TestCase):
    def test_reused_identifiers_cannot_cross_transmitters(self):
        stamp = 1550055741054
        hawthorns = (51.4582980, -2.6038980)
        def receipt(mac, position, seq, offset):
            return (f"RxMAC;{mac};109;RX-REQ-CAM;Validation;1;StationID;42;SeqNum;{seq};"
                    f"GpsLat;{position[0]};GpsLon;{position[1]};Timestamp;{stamp+offset};\n")

        other = receipt("4c:5e:0c:84:36:17", (51.4571309, -2.6128779), 0, 11)
        correct = receipt("4c:5e:0c:84:35:fb", hawthorns, 1, 20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vehicle.log"
            path.write_text(other + correct + receipt("4c:5e:0c:84:35:fb", hawthorns, 1, 14))
            heard = read(path, "13022019", "")[4]
            path.write_text(receipt("4c:5e:0c:84:35:fb", (51.4571309, -2.6128779), 1, 14))
            with self.assertRaisesRegex(RuntimeError, "position disagree"):
                read(path, "13022019", "")

        fixes = [(stamp + delta, None, *hawthorns) for delta in (-50, 50)]
        pair = [(fixes, None, None, None, heard), (fixes, None, None, None, {})]
        offers = [(stamp + seq, (42, seq), *hawthorns) for seq in (0, 1)]
        offsets = [[], []]
        self.assertEqual(list(broadcast_rows(offers, pair, offsets)),
                         [(stamp, 0, [0., 0.]), (stamp + 1, 1, [0., 0.])])
        self.assertEqual(offsets, [[13], []])
